import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import time

from utils import Optimer, compare_model_grad, compare_model_weight


torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class AttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_attn, w_proj, b_attn, b_proj, bias, masked_bias, num_head):
        B, M, N = x.shape
        H = num_head
        D_attn = w_attn.shape[1]
        assert w_attn.shape == (N, D_attn)
        assert w_proj.shape == (D_attn//3, N)  # TODO
        if b_attn != None:
            assert b_attn.shape == (D_attn,)
        if b_proj != None:
            assert b_proj.shape == (D_attn//3,)

        q = torch.empty((B, M, D_attn//3), dtype = x.dtype, device = x.device)
        k = torch.empty((B, M, D_attn//3), dtype = x.dtype, device = x.device)
        v = torch.empty((B, M, D_attn//3), dtype = x.dtype, device = x.device)

        softmax_out = torch.empty((B, H, M, M), dtype = x.dtype, device = x.device)
        soft_v = torch.empty((B, H, M, D_attn//(3*H)), dtype = x.dtype, device = x.device)
        out = torch.empty((B, M, N), dtype = x.dtype, device = x.device)

        torch.ops.my_ops.attn_forward(x,
                                     w_attn,
                                     w_proj,
                                     b_attn,
                                     b_proj,
                                     q,
                                     k,
                                     v,
                                     softmax_out,
                                     soft_v,
                                     out)

        ctx.save_for_backward(x, w_attn, w_proj, q, k, v, softmax_out, soft_v, bias)

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w_attn, w_proj, q, k, v, softmax_out, soft_v, bias = ctx.saved_tensors
        B, M, N = x.shape
        H = softmax_out.shape[1]
        D_attn = w_attn.shape[1]

        grad_x = torch.ones(x.shape, dtype=x.dtype, device = grad_output.device)
        grad_w_attn = torch.ones(w_attn.shape, dtype=x.dtype, device = grad_output.device)
        grad_w_proj = torch.ones(w_proj.shape, dtype=x.dtype, device = grad_output.device)

        grad_b_attn = grad_b_proj = None
        if ctx.needs_input_grad[3]:
            grad_b_attn = torch.ones((D_attn,), dtype=x.dtype, device = grad_output.device)
        if ctx.needs_input_grad[4]:
            grad_b_proj = torch.ones((D_attn//3,), dtype=x.dtype, device = grad_output.device)

        torch.ops.my_ops.attn_backward(grad_output,
                                        x,
                                        w_attn,
                                        w_proj,
                                        q, 
                                        k, 
                                        v,
                                        softmax_out,
                                        soft_v,
                                        bias,
                                        grad_x,
                                        grad_w_attn,
                                        grad_w_proj,
                                        grad_b_attn,
                                        grad_b_proj)

        return grad_x, grad_w_attn, grad_w_proj, grad_b_attn, grad_b_proj, None, None, None


class AttentionBlock(nn.Module):
    def __init__(self, w_attn, w_proj, b_attn, b_proj, bias, masked_bias, num_head):
        super().__init__()
        self.w_attn = w_attn
        self.w_proj = w_proj
        self.b_attn = b_attn
        self.b_proj = b_proj
        self.bias = bias
        self.masked_bias = masked_bias
        self.num_head = num_head

    def forward(self, x):
        return AttnFunc.apply(x, self.w_attn, self.w_proj, self.b_attn, self.b_proj, self.bias, self.masked_bias, self.num_head)


class Config:
    def __init__(self):
        pass


def check_attention():
    batch_size = 4
    length = 10
    config = Config
    config.max_position_embeddings = 100
    config.hidden_size = 32
    config.num_attention_heads = 8
    config.is_cross_attention = False
    net_cpu = GPT2Attention(config)
    w_attn = net_cpu.state_dict()['c_attn.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    w_proj = net_cpu.state_dict()['c_proj.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    b_attn = None
    b_proj = None
    if 'c_attn.bias' in net_cpu.state_dict().keys():
        b_attn = net_cpu.state_dict()['c_attn.bias'].clone().detach().requires_grad_(True).to(device).half()
    if 'c_proj.bias' in net_cpu.state_dict().keys():
        b_proj = net_cpu.state_dict()['c_proj.bias'].clone().detach().requires_grad_(True).to(device).half()
    bias = net_cpu.state_dict()['bias'].clone().detach().to(device).half()
    masked_bias = net_cpu.state_dict()['masked_bias'].clone().detach().to(device).half()

    net_tpu = AttentionBlock(w_attn, w_proj, b_attn, b_proj, bias, masked_bias, config.num_attention_heads)

    print("=====forward======")
    x = torch.randn(batch_size, length, config.hidden_size, requires_grad=True)
    x_tpu = x.to(device).half()
    # import pdb;pdb.set_trace()
    out_tpu = net_tpu(x_tpu)
    out_cpu = net_cpu(x)
    out_diff = out_cpu - out_tpu.float().to("cpu")
    print (torch.max(abs(out_diff)))
    import pdb;pdb.set_trace()

    print("=====backward======")
    ref_tpu = torch.ones(batch_size, length, b_proj.shape[0]).to(device)
    ref_cpu = ref_tpu.cpu()
    out_tpu.backward(ref_tpu)
    out_cpu.backward(ref_cpu)

    return


if __name__ == '__main__':
    check_attention()