import torch
import torch.nn as nn
from typing import Optional

class LLamaAttentionQKVFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                query_states,
                key_states,
                value_states,
                cos,
                sin,
                attention_mask,
                softmax_scale,
                attention_dropout):
        ctx.softmax_scale = softmax_scale
        batch, seq_len, num_attn_head, head_dim = query_states.size()
        num_kv_head = key_states.size(-2)

        cos = cos.view(seq_len, 1, head_dim)
        sin = sin.view(seq_len, 1, head_dim)
        query_states = query_states.view(batch*seq_len, num_attn_head, head_dim)
        key_states = key_states.view(batch*seq_len, num_kv_head, head_dim)
        value_states = value_states.view(batch*seq_len, num_kv_head, head_dim)

        if attention_mask.dim() == 4 and attention_mask.size(1)==1:
            attention_mask = attention_mask.squeeze(1)

        attn_output = torch.empty(query_states.shape, dtype = query_states.dtype, device = query_states.device)
        softmax_lse = torch.empty([batch*seq_len, num_attn_head, 1], dtype=torch.float32, device=query_states.device)
        input_length = torch.tensor([seq_len] * batch, dtype=torch.int32)
        max_s = input_length.max().item()

        torch.ops.my_ops.llama_attention_forward(attn_output,
                                    query_states,
                                    key_states,
                                    value_states,
                                    cos,
                                    sin,
                                    attention_mask,
                                    softmax_lse,
                                    input_length,
                                    max_s,
                                    softmax_scale,
                                    attention_dropout,
                                    len(input_length))
        attn_output = attn_output.reshape(batch, seq_len, num_attn_head, head_dim)
        softmax_lse = softmax_lse.reshape(batch, seq_len, num_attn_head, 1)
        ctx.save_for_backward(query_states, key_states, value_states, cos, sin, softmax_lse, attn_output, attention_mask)
        return attn_output

    @staticmethod
    def backward(ctx, grad_output):
        query_states, key_states, value_states, cos, sin, lse, attn_output, attention_mask = ctx.saved_tensors
        batch, seq_len, num_heads, head_dim = attn_output.size()
        num_key_value_heads = key_states.shape[-2]

        query_states = query_states.view(batch * seq_len, num_heads, head_dim)
        key_states = key_states.view(batch * seq_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch * seq_len, num_key_value_heads, head_dim)
        attn_output = attn_output.view(batch * seq_len, num_heads, head_dim)
        grad_output = grad_output.view(batch * seq_len, num_heads, head_dim)
        lse = lse.view(batch * seq_len, num_heads, 1)
        cos = cos.view(seq_len, 1, head_dim)
        sin = sin.view(seq_len, 1, head_dim)
        attention_mask = attention_mask.view(batch * seq_len, seq_len)

        grad_query_states = torch.zeros_like(query_states)
        grad_key_states = torch.zeros_like(key_states)
        grad_value_states = torch.zeros_like(value_states)

        input_lengths = torch.tensor([seq_len] * batch, dtype=torch.int32, device=query_states.device)

        torch.ops.my_ops.llama_attention_backward(
            query_states,
            key_states,
            value_states,
            attn_output,
            grad_output,
            lse,
            grad_query_states,
            grad_key_states,
            grad_value_states,
            cos,
            sin,
            attention_mask,
            input_lengths,
            seq_len,
            ctx.softmax_scale)
        grad_query_states = grad_query_states.reshape(batch, seq_len, num_heads, head_dim)
        grad_key_states = grad_key_states.reshape(batch, seq_len, num_key_value_heads, head_dim)
        grad_value_states = grad_value_states.reshape(batch, seq_len, num_key_value_heads, head_dim)
        return grad_query_states, grad_key_states, grad_value_states, None, None, None, None, None

llama_attn_qkv_fn = LLamaAttentionQKVFunc.apply

class LlamaAttentionQKV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                query_states,
                key_states,
                value_states,
                cos,
                sin,
                attention_mask,
                softmax_scale,
                attention_dropout = 0.0):
        return llama_attn_qkv_fn(
            query_states,
            key_states,
            value_states,
            cos,
            sin,
            attention_mask,
            softmax_scale,
            attention_dropout)

def init_wrapper(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.attn_qkv = LlamaAttentionQKV()
    return wrapper

def llama_attn_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
    bsz, q_len, _ = hidden_states.size()

    if(not hasattr(self, 'attn_qkv')):
        self.attn_qkv = LlamaAttentionQKV()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    cos, sin = self.rotary_emb(value_states, position_ids)
    softmax_scale = 1.0 / self.head_dim ** 0.5
    attn_output = self.attn_qkv(
        query_states,
        key_states,
        value_states,
        cos,
        sin,
        attention_mask,
        softmax_scale,
        self.attention_dropout)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def fuse_llama_attn_qkv():
    import transformers
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward
    transformers.models.llama.modeling_llama.LlamaAttention.__init__ = init_wrapper(transformers.models.llama.modeling_llama.LlamaAttention.__init__)


def qwen2_attn_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None, #past_key_value: Optional[Cache] = None
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
    bsz, q_len, _ = hidden_states.size()

    if(not hasattr(self, 'attn_qkv')):
        self.attn_qkv = LlamaAttentionQKV()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[1])
    softmax_scale = 1.0 / self.head_dim ** 0.5
    attn_output = self.attn_qkv(
        query_states,
        key_states,
        value_states,
        cos,
        sin,
        attention_mask,
        softmax_scale,
        self.attention_dropout)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def fuse_qwen2_attn_qkv():
    import transformers
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attn_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__init__ = init_wrapper(transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__init__)