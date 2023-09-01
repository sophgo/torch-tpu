import torch
import torch.nn as nn
import torch.nn.functional as F
import os

torch.ops.load_library("../../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

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

        ctx.save_for_backward(x, w, mean, rstd)

        return mean, rstd, out 


class lnMatmulBlock(nn.Module):
    def __init__(self, w, b, gamma, beta):
        super().__init__()
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta

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

        ctx.save_for_backward(x1, x2, w, mean, rstd)

        return out_add, mean, rstd, out 


class AddlnMatmulBlock(nn.Module):
    def __init__(self, w, b, gamma, beta):
        super().__init__()
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta

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
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()

            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        attn_weights = self.attn_dropout(attn_weights)
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
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # x = self.c_attn(hidden_states)

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

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)

        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # hidden_states = self.c_fc(hidden_states)

        hidden_states = self.act(hidden_states)

        hidden_states = self.c_proj(hidden_states)

        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.embed_dim = config.hidden_size
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.c_fc = Conv1D(inner_dim, config.hidden_size)
        self.mlp = GPT2MLP(inner_dim, config)

        self.ln_fc = lnMatmulBlock(self.c_attn.weight, self.c_attn.bias, self.ln_1.weight, self.ln_1.bias)
        self.add_ln_fc = AddlnMatmulBlock(self.c_fc.weight, self.c_fc.bias, self.ln_2.weight, self.ln_2.bias)

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
        _, _, x_ = self.ln_fc(hidden_states)
        hidden_states = self.ln_1(hidden_states)
        x = self.c_attn(hidden_states)
        print(torch.max(abs(x_.cpu()-x.cpu())))
        import pdb;pdb.set_trace()
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

        residual_, _, _, hidden_states_ = self.add_ln_fc(attn_output, residual)

        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.c_fc(hidden_states)

        print(torch.max(abs(hidden_states_.cpu()-hidden_states.cpu())))
        print(torch.max(abs(residual_.cpu()-residual.cpu())))
        import pdb;pdb.set_trace()

        feed_forward_hidden_states = self.mlp(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

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
    configure.n_positions = 64
    configure.n_embd = 128
    configure.n_head = 16
    configure.n_layer = 1

    batch = 2
    sequence = 32
    inp = torch.rand(batch, sequence, configure.hidden_size).to(device).half()
    ref = torch.ones((batch, sequence, configure.hidden_size)).to(device).half()

    net = GPT2Block(configure).to(device).half()

    out_cpu = net(inp)

    # out_cpu[0].backward(ref)

    
