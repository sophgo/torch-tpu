import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from utils import ForwardHack, BackwardHack,DumpIns
torch.set_num_threads(1)

DI = DumpIns()
torch.manual_seed(1000)
TP = 16


class lnMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, gamma, beta):
        D = w.shape[1]
        if x.dim() == 3:
            B, M, N = x.shape
            
            mean = torch.empty((B, M), dtype = x.dtype, device = x.device)
            rstd = torch.empty((B, M), dtype = x.dtype, device = x.device)
            out = torch.empty((B, M, D), dtype = x.dtype, device = x.device)
        else:
            M, N = x.shape
            
            mean = torch.empty((M,), dtype = x.dtype, device = x.device)
            rstd = torch.empty((M,), dtype = x.dtype, device = x.device)
            out = torch.empty((M, D), dtype = x.dtype, device = x.device)
        assert w.shape == (N, D)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        if b != None:
            assert b.shape == (D,)

        torch.ops.my_ops.ln_mm_forward(x,
                                     w,
                                     b,
                                     gamma,
                                     beta,
                                     1e-5, 
                                     mean,
                                     rstd,
                                     out)

        ctx.save_for_backward(x, w, mean, rstd, gamma, beta)

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, mean, rstd, gamma, beta = ctx.saved_tensors
        D = w.shape[1]
        if x.dim() == 3:
            B, M, N = x.shape
            out_ln_cpu = ((x.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).unsqueeze(1).cpu() + beta.unsqueeze(0).unsqueeze(1).cpu()
            grad_out_ln = torch.matmul(grad_output, w.unsqueeze(0).transpose(-1,-2))
        else:
            M, N = x.shape
            out_ln_cpu = ((x.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).cpu() + beta.unsqueeze(0).cpu()
            grad_out_ln = torch.matmul(grad_output, w.transpose(-1,-2))

        out_ln = out_ln_cpu.to(grad_output.device)

        grad_x = torch.ones(x.shape, dtype = x.dtype, device = grad_output.device)
        grad_gamma = torch.ones((N,), dtype = x.dtype, device = grad_output.device)
        grad_beta = torch.ones((N,), dtype = x.dtype, device = grad_output.device)

        grad_w = torch.matmul(out_ln.transpose(-1,-2), grad_output)
        grad_b = grad_output.reshape(-1, D).sum(0)
        
        torch.ops.my_ops.ln_mm_backward(grad_out_ln,
                                        x,
                                        mean.unsqueeze(-1),
                                        rstd.unsqueeze(-1),
                                        gamma,
                                        grad_x,
                                        grad_gamma,
                                        grad_beta)
        
        return grad_x, grad_w, grad_b, grad_gamma, grad_beta 


class lnMatmulBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        w = torch.empty(in_dim, out_dim)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(out_dim))

        gamma = torch.ones(in_dim,)
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return lnMatmulFunc.apply(x, self.w, self.b, self.gamma, self.beta)
    

class AddlnMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, w, b, gamma, beta):
        D = w.shape[1]

        if x1.dim() == 3:
            B, M, N = x1.shape
            mean = torch.empty((B, M), dtype = x1.dtype, device = x1.device)
            rstd = torch.empty((B, M), dtype = x1.dtype, device = x1.device)
            out = torch.empty((B, M, D), dtype = x1.dtype, device = x1.device)
        else:
            M, N = x1.shape
            mean = torch.empty((M,), dtype = x1.dtype, device = x1.device)
            rstd = torch.empty((M,), dtype = x1.dtype, device = x1.device)
            out = torch.empty((M, D), dtype = x1.dtype, device = x1.device)
        out_add = torch.empty(x1.shape, dtype = x1.dtype, device = x1.device)

        assert x1.shape == x2.shape
        assert w.shape == (N, D)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        if b != None:
            assert b.shape == (D,)

        torch.ops.my_ops.add_ln_mm_forward(x1,
                                     x2,
                                     w,
                                     b,
                                     gamma,
                                     beta,
                                     1e-5,
                                     out_add,
                                     mean,
                                     rstd,
                                     out)

        ctx.save_for_backward(out_add, w, mean, rstd, gamma, beta)

        return out_add, out
    
    @staticmethod
    def backward(ctx, grad_add, grad_output):
        out_add, w, mean, rstd, gamma, beta = ctx.saved_tensors

        D = w.shape[1]
        if out_add.dim() == 3:
            B, M, N = out_add.shape
            out_ln_cpu = ((out_add.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).unsqueeze(1).cpu() + beta.unsqueeze(0).unsqueeze(1).cpu()
            grad_out_ln = torch.matmul(grad_output, w.unsqueeze(0).transpose(-1,-2))
        else:
            M, N = out_add.shape
            out_ln_cpu = ((out_add.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).cpu() + beta.unsqueeze(0).cpu()
            grad_out_ln = torch.matmul(grad_output, w.transpose(-1,-2))

        out_ln = out_ln_cpu.to(grad_output.device)

        grad_out_add = torch.ones(out_add.shape, dtype = out_add.dtype, device = grad_output.device)
        grad_gamma = torch.ones((N,), dtype = out_add.dtype, device = grad_output.device)
        grad_beta = torch.ones((N,), dtype = out_add.dtype, device = grad_output.device)

        grad_w = torch.matmul(out_ln.transpose(-1,-2), grad_output)
        grad_b = grad_output.reshape(-1, D).sum(0)
        
        torch.ops.my_ops.add_ln_mm_backward(grad_out_ln,
                                        out_add,
                                        mean.unsqueeze(-1),
                                        rstd.unsqueeze(-1),
                                        gamma,
                                        grad_out_add,
                                        grad_gamma,
                                        grad_beta)
        
        return grad_out_add, grad_out_add, grad_w, grad_b, grad_gamma, grad_beta


class AddlnMatmulBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        w = torch.empty(in_dim, out_dim)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(out_dim))

        gamma = torch.ones(in_dim,)
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(torch.zeros(in_dim))



    def forward(self, x1, x2):
        return AddlnMatmulFunc.apply(x1, x2, self.w, self.b, self.gamma, self.beta)



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


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        DI.dump("Matmul_QK")
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
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
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = BackwardHack.apply("SOFTMAX_QK_d_", attn_weights)

        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        DI.dump("Matmul_QKV")
        attn_output = torch.matmul(attn_weights, value)
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
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # if encoder_hidden_states is not None:
        #     if not hasattr(self, "q_attn"):
        #         raise ValueError(
        #             "If class is used as cross attention, the weights `q_attn` have to be defined. "
        #             "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
        #         )

        #     query = self.q_attn(hidden_states)
        #     key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        #     attention_mask = encoder_attention_mask
        # else:
        #     DI.dump("FC_QKV")
        #     x = self.c_attn(hidden_states)
        #     x = BackwardHack.apply("FC_QKV_dwi_", x)

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
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

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
        self.act = F.gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # DI.dump("FC_mlp0")
        # hidden_states = self.c_fc(hidden_states)
        # hidden_states = BackwardHack.apply("FC_mlp0_dwi_", hidden_states)

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

        self.embed_dim = config.hidden_size
        # self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        # self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # if config.add_cross_attention:
        #     self.crossattention = GPT2Attention(config, is_cross_attention=True)
        #     self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

        self.ln_fc = lnMatmulBlock(self.embed_dim, 3 * self.embed_dim // TP)
        self.add_ln_fc = AddlnMatmulBlock(config.hidden_size, inner_dim // TP)

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
        residual = hidden_states
        DI.dump("LayerNorm_FC")
        x = self.ln_fc(hidden_states)
        x = BackwardHack.apply("LayerNorm_FC_", x)
        # DI.dump("LayerNorm_embedded_input")
        # hidden_states = self.ln_1(hidden_states)
        # hidden_states = BackwardHack.apply("LayerNorm_embedded_input_drbi_", hidden_states)

        attn_outputs = self.attn(
            x,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        #t4 = time.time()
        #print("attetion time", t4 - t3)
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        DI.dump("Add_LayerNorm_FC")
        residual, hidden_states = self.add_ln_fc(attn_output, residual)
        hidden_states = BackwardHack.apply("Add_LayerNorm_FC_", hidden_states)

        # DI.dump("Add_atten")
        # hidden_states = attn_output + residual
        # hidden_states = BackwardHack.apply("Add_atten_", hidden_states)

        # if encoder_hidden_states is not None:
        #     # add one self-attention block for cross-attention
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
        #             "cross-attention layers by setting `config.add_cross_attention=True`"
        #         )
        #     residual = hidden_states
        #     hidden_states = self.ln_cross_attn(hidden_states)
        #     cross_attn_outputs = self.crossattention(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         output_attentions=output_attentions,
        #     )
        #     attn_output = cross_attn_outputs[0]
        #     # residual connection
        #     hidden_states = residual + attn_output
        #     outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # residual = hidden_states
        # DI.dump("LayerNorm_atten")
        # hidden_states = self.ln_2(hidden_states)
        # hidden_states = BackwardHack.apply("LayerNorm_atten_drbi_", hidden_states)

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
    # configure = GPT2Config()
    # configure.attn_pdrop = 0
    # configure.embd_pdrop = 0
    # configure.resid_pdrop = 0
    # configure.activation_function= "gelu"
    # configure.n_positions = 4096
    # configure.n_embd = 12288
    # configure.n_head = 96
    # configure.n_layer = 1

    # batch = 6
    # sequence = 4096
    ########################################
    configure = GPT2Config()
    configure.attn_pdrop = 0
    configure.embd_pdrop = 0
    configure.resid_pdrop = 0
    configure.activation_function= "gelu"
    # configure.n_positions = 16
    # configure.n_embd = 256
    # configure.n_head = 16
    # configure.n_layer = 1

    # batch = 2
    # sequence = 16

    configure.n_positions = 4096
    configure.n_embd = 12288
    configure.n_head = 96
    configure.n_layer = 1

    batch = 6
    sequence = 4096
    
    DI.dump("Copy_input")
    inp = torch.rand(batch, sequence, configure.hidden_size).to("tpu:0").half()
    ref = torch.ones((batch, sequence, configure.hidden_size)).to("tpu:0").half()

    DI.dump("Copy_model_weight")
    net = GPT2Block(configure).to("tpu:0").half()

    out_cpu = net(inp)

    out_cpu[0].backward(ref)

