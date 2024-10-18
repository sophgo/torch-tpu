import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from utils import ForwardHack, BackwardHack,DumpIns, Dump_Data, TensorComparator
from decorator import register_hook
import sys

torch.set_num_threads(1)

DI = DumpIns()
torch.manual_seed(1000)
TP = 16

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return F.gelu(x)

class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, left, right):
        return torch.matmul(left, right)

class SoftMax(nn.Module):
    def __init__(self, dim=None):
        super(SoftMax, self).__init__()

    def forward(self, x, dim):
        return nn.Softmax(dim=dim)(x)
    
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

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
        self.num_heads = config.num_attention_heads // TP
        self.head_dim = self.embed_dim // TP // self.num_heads
        self.split_size = self.embed_dim // TP
        if self.head_dim * self.num_heads * TP != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim // TP, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim//TP)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        self.matmul = MatMul()
        self.softmax = SoftMax()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, device='cpu'):
        DI.dump("Matmul_QK")
        attn_weights = self.matmul(query, key.transpose(-1, -2))
        attn_weights = BackwardHack.apply("Matmul_QK_dqk_", attn_weights)

        if self.scale_attn_weights:
            DI.dump("Norm_QK[Div]")
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
            attn_weights = BackwardHack.apply("Norm_QK[Div]_d_", attn_weights)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            DI.dump("Get_QK_MASK")
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            causal_mask = BackwardHack.apply("Get_QK_MASK_", causal_mask)

            DI.dump("WHERE_ON_QK")
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
            attn_weights = BackwardHack.apply("WHERE_ON_dqkv_", attn_weights)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        
        DI.dump("SOFTMAX_QK")
        attn_weights = self.softmax(attn_weights, dim=-1)
        attn_weights = BackwardHack.apply("SOFTMAX_QK_d_", attn_weights)

        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        DI.dump("Matmul_QKV")
        attn_output = self.matmul(attn_weights, value)
        attn_output = BackwardHack.apply("Matmul_QKV_dqkv_", attn_output)

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
        device='cpu',
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
            DI.dump("FC_QKV")
            x = self.c_attn(hidden_states)
            x = BackwardHack.apply("FC_QKV_dwi_", x)

            query, key, value = x.split(self.split_size, dim=2)

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
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, device)

        DI.dump("Concat_heads")
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = BackwardHack.apply("Concat_heads_d_", attn_output)


        DI.dump("FC_proj")
        attn_output = self.c_proj(attn_output)
        attn_output = BackwardHack.apply("FC_proj_dwi_", attn_output)

        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size//TP, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size//TP)
        self.act = GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        DI.dump("FC_mlp0")
        hidden_states = self.c_fc(hidden_states)
        hidden_states = BackwardHack.apply("FC_mlp0_dwi_", hidden_states)

        DI.dump("GeLU")
        hidden_states = self.act(hidden_states)
        hidden_states = BackwardHack.apply("gelu_dx_", hidden_states)

        DI.dump("FC_mlp1")
        hidden_states = self.c_proj(hidden_states)
        hidden_states = BackwardHack.apply("FC_mlp1_dwi_", hidden_states)

        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        self.device = 'cpu'
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
        device='cpu',
    ):
        residual = hidden_states
        DI.dump("LayerNorm_embedded_input")
        hidden_states = self.ln_1(hidden_states)
        hidden_states = BackwardHack.apply("LayerNorm_embedded_input_drbi_", hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            device = device,
        )
        #t4 = time.time()
        #print("attetion time", t4 - t3)
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        DI.dump("Add_atten")
        hidden_states = attn_output + residual
        hidden_states = BackwardHack.apply("Add_atten_", hidden_states)

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        DI.dump("LayerNorm_atten")
        hidden_states = self.ln_2(hidden_states)
        hidden_states = BackwardHack.apply("LayerNorm_atten_drbi_", hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        #t2 = time.time()
        #print("mlp time", t2 - t1)

        # residual connection
        DI.dump("Add_layer_result")
        hidden_states = residual + feed_forward_hidden_states
        hidden_states = BackwardHack.apply("Add_layer_result_d_", hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        #import pdb;pdb.set_trace()
        return outputs  # hidden_states, present, (attentions, cross_attentions)

if __name__ == "__main__":
    from transformers import GPT2Config
    import copy
    import time
    ############# configure ###############
    configure = GPT2Config()
    configure.attn_pdrop = 0
    configure.embd_pdrop = 0
    configure.resid_pdrop = 0
    configure.activation_function= "gelu"
    configure.n_positions = 4096
    configure.n_embd = 12288
    configure.n_head = 96
    configure.n_layer = 1

    batch = 6
    sequence = 4096
    ########################################
    
    DI.dump("Copy_input")
    inp = torch.rand(batch, sequence, configure.hidden_size)
    ref = torch.ones((batch, sequence, configure.hidden_size))

    inp_tpu = copy.deepcopy(inp).to("tpu:0").half()
    ref_tpu = copy.deepcopy(ref).to("tpu:0").half()

    DI.dump("Copy_model_weight")
    net = GPT2Block(configure)
    net_tpu = copy.deepcopy(net)
    net_tpu = net_tpu.to("tpu:0").half()

    # Use hook if want to dump di/do/dw 
    # net_tpu.apply(lambda net: register_hook(net, 'tpu'))
    # net.apply(lambda net: register_hook(net, 'cpu'))

    print("============ Forward ============")
    out_cpu = net(inp, device='cpu')
    out_tpu = net_tpu(inp_tpu, device='tpu')
    # print("cpu",out_cpu[0].cpu())
    # print("tpu",out_tpu[0].cpu())

    print("============ Forward compare result start ============")
    comparator = TensorComparator()
    status = comparator.cmp_result(out_cpu[0].detach(), out_tpu[0].cpu().detach().float())
    if status == -1 :
        print(f"Forward output compare failed")
        sys.exit(255)
    print("============ Forward compare success ============")

    print("============ Backward ============")
    out_cpu[0].backward(ref)
    out_tpu[0].backward(ref_tpu)

    print("============ Backward compare result ============")

    grad_weight_dict = {
        'CPU' : {
            'ln_1_weight_grad': net.ln_1.weight.grad.cpu(),
            'attn_c_attn_weight_grad': net.attn.c_attn.weight.grad.cpu(),
            'attn_c_proj_weight_grad': net.attn.c_proj.weight.grad.cpu(),
            'ln_2_weight_grad': net.ln_2.weight.grad.cpu(),
            'mlp_c_fc_weight_grad': net.mlp.c_fc.weight.grad.cpu(),
            'mlp_c_proj_weight_grad': net.mlp.c_proj.weight.grad.cpu()
        },
        'TPU' : {
            'ln_1_weight_grad': net_tpu.ln_1.weight.grad.cpu(),
            'attn_c_attn_weight_grad': net_tpu.attn.c_attn.weight.grad.cpu(),
            'attn_c_proj_weight_grad': net_tpu.attn.c_proj.weight.grad.cpu(),
            'ln_2_weight_grad': net_tpu.ln_2.weight.grad.cpu(),
            'mlp_c_fc_weight_grad': net_tpu.mlp.c_fc.weight.grad.cpu(),
            'mlp_c_proj_weight_grad': net_tpu.mlp.c_proj.weight.grad.cpu()
        }
    }

    for tensor_name in grad_weight_dict['TPU']:
        print(f"============ compare {tensor_name} ============")
        grad_tpu = grad_weight_dict['TPU'][tensor_name]
        grad_cpu = grad_weight_dict['CPU'][tensor_name]
        status = comparator.cmp_result(grad_cpu, grad_tpu.float())
        if status == -1 :
            print(f"{tensor_name} compare failed")
            sys.exit(255)
    print("============ Backward compare success ============")
    print("===== All test success =====")

    # for key, value in weight_tpu_dict.items() :
    #     Dump_Data(value, key, "tpu")
    # for key, value in weight_cpu_dict.items() :
    #     Dump_Data(value, key, "cpu")
