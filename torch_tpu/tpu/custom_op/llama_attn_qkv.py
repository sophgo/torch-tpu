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

        cos = cos.repeat(batch, 1, 1).view(batch*seq_len, 1, head_dim)
        sin = sin.repeat(batch, 1, 1).view(batch*seq_len, 1, head_dim)
        query_states = query_states.view(batch*seq_len, num_attn_head, head_dim)
        key_states = key_states.view(batch*seq_len, num_kv_head, head_dim)
        value_states = value_states.view(batch*seq_len, num_kv_head, head_dim)

        if attention_mask.dim() == 4 and attention_mask.size(1)==1:
            attention_mask = attention_mask.squeeze(1)

        attn_output = torch.empty(query_states.shape, dtype = query_states.dtype, device = query_states.device)
        softmax_lse = torch.empty([batch*seq_len, num_attn_head, 1], dtype=query_states.dtype, device=query_states.device)
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
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        query_states =query_states.view(batch, seq_len, num_heads, head_dim)
        key_states =key_states.view(batch, seq_len, num_key_value_heads, head_dim)
        value_states =value_states.view(batch, seq_len, num_key_value_heads, head_dim)

        grad_query_states = torch.zeros_like(query_states)
        grad_key_states = torch.zeros_like(key_states)
        grad_value_states = torch.zeros_like(value_states)

        slices = min(batch, 4096 // seq_len)
        secs = (batch + slices - 1) // slices
        slices = (batch + secs - 1) // secs
        for i in range(0, batch, slices):
            real_slices = min(slices, batch - i)
            query_states_batch = query_states[i: i + real_slices].view(real_slices * seq_len, num_heads, head_dim)
            key_states_batch = key_states[i: i + real_slices].view(real_slices * seq_len, num_key_value_heads, head_dim)
            value_states_batch = value_states[i: i + real_slices].view(real_slices * seq_len, num_key_value_heads, head_dim)
            attn_output_batch = attn_output[i: i + real_slices].view(real_slices * seq_len, num_heads, head_dim)
            grad_output_batch = grad_output[i: i + real_slices].view(real_slices * seq_len, num_heads, head_dim)
            lse_batch = lse[i: i + real_slices].view(real_slices * seq_len, num_heads, 1)
            cos_batch = cos[i* seq_len: (i + real_slices)* seq_len].view(real_slices * seq_len, 1, head_dim)
            sin_batch = sin[i* seq_len: (i + real_slices)* seq_len].view(real_slices * seq_len, 1, head_dim)
            if attention_mask is not None:
                if real_slices == 1:
                    attention_mask_batch = attention_mask[0][0]
                else:
                    attention_mask_batch = torch.full((real_slices * seq_len, real_slices * seq_len), float("-inf"), dtype=attention_mask.dtype, device=attention_mask.device)
                    for j in range(real_slices):
                        attention_mask_batch[j * seq_len: (j + 1) * seq_len, j * seq_len: (j + 1) * seq_len] = attention_mask[i + j][0]
            else:
                attention_mask_batch = None
            grad_query_states_batch = grad_query_states[i: i + real_slices].view(real_slices * seq_len, num_heads, head_dim)
            grad_key_states_batch = grad_key_states[i: i + real_slices].view(real_slices * seq_len, num_key_value_heads, head_dim)
            grad_value_states_batch = grad_value_states[i: i + real_slices].view(real_slices * seq_len, num_key_value_heads, head_dim)
            input_lengths = torch.tensor([seq_len * real_slices], dtype=torch.int32, device=query_states.device)

            torch.ops.my_ops.llama_attention_backward(
                query_states_batch,
                key_states_batch,
                value_states_batch,
                attn_output_batch,
                grad_output_batch,
                lse_batch,
                grad_query_states_batch,
                grad_key_states_batch,
                grad_value_states_batch,
                cos_batch,
                sin_batch,
                attention_mask_batch,
                input_lengths,
                real_slices * seq_len,
                ctx.softmax_scale)

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