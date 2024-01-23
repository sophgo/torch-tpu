import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys 
sys.path.append("..") 
from utils import get_model_grad, Optimer, compare_model_grad
from transformers import GPT2Config
import copy
import time
torch.manual_seed(1000)
torch.ops.load_library("../../../build/torch_tpu/libtorch_tpu.so")
optimer = Optimer("../../../build/torch_tpu/libtorch_tpu.so")

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
        t1 = time.time()
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        t2 = time.time()
        print("====inner_attn matmul-qk time ", t2 - t1)
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            t1 = time.time()
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
            t2 = time.time()
            print("====inner_attn where time ", t2 - t1)            

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        #import pdb;pdb.set_trace()
        t1 = time.time()
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        t2 = time.time()
        print("====inner_attn softmax time ", t2 - t1)
        
        t1 = time.time()
        attn_weights = self.attn_dropout(attn_weights)
        t2 = time.time()
        print("====inner_attn attn_dropout time ", t2 - t1)

        # Mask heads if we want to
        if head_mask is not None:
            t1 = time.time()
            attn_weights = attn_weights * head_mask
            t2 = time.time()
            print("====inner_attn binary-mul attn time ", t2 - t1)

        t1 = time.time()
        attn_output = torch.matmul(attn_weights, value)
        t2 = time.time()
        print("====inner_attn batch-mm time ", t2 - t1)

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
        
        t1 = time.time()
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        t2 = time.time()
        print("split time", t2 - t1)

        t1 = time.time()
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        t2 = time.time()
        print("cat time", t2 - t1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        t1 = time.time()
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        t2 = time.time()
        print("inner_att time", t2 - t1)

        t1 = time.time()
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        t2 = time.time()
        print("permute time", t2 - t1)
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
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
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
        t1 = time.time()
        hidden_states = self.ln_1(hidden_states)
        t2 = time.time()
        print("layernorm1 time ", t2 - t1)
        t3 = time.time()
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        t4 = time.time()
        print("attetion time", t4 - t3)
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

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

        t1 = time.time()
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        t2 = time.time()
        print("layernorm time", t2 - t1)

        t1 = time.time()
        feed_forward_hidden_states = self.mlp(hidden_states)
        t2 = time.time()
        print("mlp time", t2 - t1)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        #import pdb;pdb.set_trace()
        return outputs  # hidden_states, present, (attentions, cross_attentions)

def case_gptblock_backward():
    ############# configure ###############
    device = torch.device("tpu:0")
    configure = GPT2Config()
    configure.attn_pdrop = 0
    configure.embd_pdrop = 0
    configure.resid_pdrop = 0
    configure.activation_function= "gelu"

    batch = 32
    sequence = 256 
    ########################################

    inp = torch.rand(batch, sequence, configure.hidden_size)
    ref = torch.rand(batch, sequence, configure.hidden_size)
    #inp = torch.ones(batch, sequence, configure.hidden_size)
    #ref = torch.ones(batch, sequence, configure.hidden_size)
    inp_tpu = inp.to(device).half()
    ref_tpu = ref.to(device).half()

    net = GPT2Block(configure)
    net_tpu = copy.deepcopy(net)
    net_tpu.to(device).half()

    print("===== forward =========")
    t1 = time.time()
    optimer.reset()
    out_tpu = net_tpu(inp_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", (t2 - t1) * 1e6)

    t1 = time.time()
    out_cpu = net(inp)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    print("===== backward =========")
    t1 = time.time()
    optimer.reset()
    out_tpu[0].backward(ref_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", (t2 - t1) * 1e6)

    t1 = time.time()
    out_cpu[0].backward(ref)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    compare_model_grad(net, net_tpu)

    def my_print(out_cpu, out_tpu):
        for i in range(len(out_cpu)):
            o_c = out_cpu[i]
            if isinstance(o_c, torch.Tensor):
                o_t = out_tpu[i].to("cpu")
                #print("cpu:")
                #print(o_c)
                #print("tpu:")
                #print(o_t)
                print("max diff:", torch.max(abs(o_c - o_t)))
            elif isinstance(o_c, tuple):
                my_print(out_cpu[i], out_tpu[i])
            else:
                return
    my_print(out_cpu, out_tpu)    

def case_gptmlp_backward():
    ############# configure ###############
    device = torch.device("tpu:0")
    configure = GPT2Config()
    configure.attn_pdrop = 0
    configure.embd_pdrop = 0
    configure.resid_pdrop = 0
    configure.n_layer= 2
    configure.activation_function= "gelu"

    batch = 32
    sequence = 256
    ########################################

    inp = torch.rand(batch, sequence, configure.hidden_size)
    ref = torch.rand(batch, sequence, configure.hidden_size)
    #inp = torch.ones(batch, sequence, configure.hidden_size)
    #ref = torch.ones(batch, sequence, configure.hidden_size)
    inp_tpu = inp.to(device).half()
    ref_tpu = ref.to(device).half()

    net = GPT2MLP(configure.hidden_size * 3, configure)
    net_tpu = copy.deepcopy(net)
    net_tpu.to(device).half()

    print("===== forward =========")
    t1 = time.time()
    optimer.reset()
    out_tpu = net_tpu(inp_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    t1 = time.time()
    out_cpu = net(inp)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    print("===== backward =========")
    t1 = time.time()
    optimer.reset()
    out_tpu.backward(ref_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    t1 = time.time()
    out_cpu.backward(ref)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    compare_model_grad(net, net_tpu)

    print(" ======== compare model's out  =======")
    def my_print(out_cpu, out_tpu):
        for i in range(len(out_cpu)):
            o_c = out_cpu[i]
            if isinstance(o_c, torch.Tensor):
                o_t = out_tpu[i].to("cpu")
                #print("cpu:")
                #print(o_c)
                #print("tpu:")
                #print(o_t)
                print("max diff:", torch.max(abs(o_c - o_t)))
            elif isinstance(o_c, tuple):
                my_print(out_cpu[i], out_tpu[i])
            else:
                return
    my_print(out_cpu, out_tpu)    


if __name__ == "__main__":
    case_gptblock_backward()
