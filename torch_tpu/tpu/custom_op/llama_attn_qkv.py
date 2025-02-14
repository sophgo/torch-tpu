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
        query_states = query_states.view(batch, seq_len, num_heads, head_dim)
        key_states = key_states.view(batch, seq_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch, seq_len, num_key_value_heads, head_dim)

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


import os
from distutils.util import strtobool
whole_net_trans = strtobool(os.environ.get("QWEN2_WHOLE_NET_TRANS", "0"))
class MegatronQwen2AttentionQKVFunc(torch.autograd.Function):
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
        # cos, sin: (seq, 1, 1, head_dim), fp32
        # attention_mask: (batch, 1, seq, seq), bool
        ctx.softmax_scale = softmax_scale
        if whole_net_trans:
            # q, k, v: (seq, batch, num_heads, head_dim)
            batch, seq_len, num_attn_head, head_dim = query_states.size()
            output_shape = (batch, seq_len, num_attn_head, head_dim)
            cos = cos.to(query_states.dtype).view(seq_len, 1, head_dim)
            sin = sin.to(query_states.dtype).view(seq_len, 1, head_dim)
            input_length = torch.tensor([seq_len] * batch, dtype=torch.int32)#TODO:construct data in device 
        else:
            # q, k, v: (batch, seq, num_heads, head_dim)
            seq_len, batch, num_attn_head, head_dim = query_states.size()
            output_shape = (seq_len, batch, num_attn_head, head_dim)
            cos = cos.to(query_states.dtype).repeat(1, batch, 1, 1).view(batch * seq_len, 1, head_dim)
            sin = sin.to(query_states.dtype).repeat(1, batch, 1, 1).view(batch * seq_len, 1, head_dim)
            input_length = torch.tensor([seq_len * batch], dtype=torch.int32)#TODO:construct data in device 
        num_kv_head = key_states.size(-2)

        query_states = query_states.view(batch * seq_len, num_attn_head, head_dim)
        key_states = key_states.view(batch * seq_len, num_kv_head, head_dim)
        value_states = value_states.view(batch * seq_len, num_kv_head, head_dim)

        if attention_mask is not None:
            if whole_net_trans:
                attention_mask_2d = attention_mask.view(batch, seq_len, seq_len)
            else:
                attention_mask_2d = torch.full((batch * seq_len, batch * seq_len), float("-inf"), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(batch):
                    attention_mask_2d[i::batch, i::batch] = attention_mask[i][0]
        else:
            attention_mask_2d = None

        attn_output = torch.empty(query_states.shape, dtype = query_states.dtype, device = query_states.device)
        softmax_lse = torch.empty([batch * seq_len, num_attn_head, 1], dtype=query_states.dtype, device=query_states.device)
        # input_length = torch.tensor([seq_len * batch], dtype=torch.int32)#TODO:construct data in device 
        max_s = input_length.max().item()

        torch.ops.my_ops.llama_attention_forward(attn_output,
                                    query_states,
                                    key_states,
                                    value_states,
                                    cos,
                                    sin,
                                    attention_mask_2d,
                                    softmax_lse,
                                    input_length,
                                    max_s,
                                    softmax_scale,
                                    attention_dropout,
                                    len(input_length))
        attn_output = attn_output.view(output_shape)
        # softmax_lse = softmax_lse.view(seq_len, batch, num_attn_head, 1)
        ctx.save_for_backward(query_states, key_states, value_states, cos, sin, softmax_lse, attn_output, attention_mask_2d)
        return attn_output

    @staticmethod
    def backward(ctx, grad_output):
        query_states, key_states, value_states, cos, sin, lse, attn_output, attention_mask = ctx.saved_tensors
        if whole_net_trans:
            batch, seq_len, num_heads, head_dim = attn_output.size()
            num_key_value_heads = key_states.shape[-2]
            
            query_states = query_states.view(batch * seq_len, num_heads, head_dim)
            key_states = key_states.view(batch * seq_len, num_key_value_heads, head_dim)
            value_states = value_states.view(batch * seq_len, num_key_value_heads, head_dim)
            attn_output = attn_output.view(batch * seq_len, num_heads, head_dim)
            grad_output = grad_output.view(batch * seq_len, num_heads, head_dim)
            lse = lse.view(batch * seq_len, num_heads, 1)
            cos = cos.view(1, seq_len, head_dim).repeat(batch, 1, 1).view(batch * seq_len, 1, head_dim)
            sin = sin.view(1, seq_len, head_dim).repeat(batch, 1, 1).view(batch * seq_len, 1, head_dim)

            grad_query_states = torch.zeros_like(query_states)
            grad_key_states = torch.zeros_like(key_states)
            grad_value_states = torch.zeros_like(value_states)

            if attention_mask is not None:
                attention_mask = attention_mask.view(batch * seq_len, seq_len)
            else:
                attention_mask = None
            
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
            
            grad_query_states = grad_query_states.view(batch, seq_len, num_heads, head_dim)
            grad_key_states = grad_key_states.view(batch, seq_len, num_key_value_heads, head_dim)
            grad_value_states = grad_value_states.view(batch, seq_len, num_key_value_heads, head_dim)

            return grad_query_states, grad_key_states, grad_value_states, None, None, None, None, None
        else:
            seq_len, batch, num_heads, head_dim = attn_output.size()
            assert seq_len * batch <= 4096, "batch size too large"
            num_key_value_heads = key_states.shape[-2]

            grad_query_states = torch.zeros_like(query_states)
            grad_key_states = torch.zeros_like(key_states)
            grad_value_states = torch.zeros_like(value_states)

            query_states = query_states.view(batch * seq_len, num_heads, head_dim)
            key_states = key_states.view(batch * seq_len, num_key_value_heads, head_dim)
            value_states = value_states.view(batch * seq_len, num_key_value_heads, head_dim)
            
            attn_output = attn_output.view(batch * seq_len, num_heads, head_dim)
            grad_output = grad_output.view(batch * seq_len, num_heads, head_dim)
            lse = lse.view(batch * seq_len, num_heads, 1)
            cos = cos.view(batch * seq_len, 1, head_dim)
            sin = sin.view(batch * seq_len, 1, head_dim)
            
            grad_query_states = grad_query_states.view(batch * seq_len, num_heads, head_dim)
            grad_key_states = grad_key_states.view(batch * seq_len, num_key_value_heads, head_dim)
            grad_value_states = grad_value_states.view(batch * seq_len, num_key_value_heads, head_dim)
            
            input_lengths = torch.tensor([seq_len * batch], dtype=torch.int32, device=query_states.device)

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
                batch * seq_len,
                ctx.softmax_scale)
            
            grad_query_states = grad_query_states.view(seq_len, batch, num_heads, head_dim)
            grad_key_states = grad_key_states.view(seq_len, batch, num_key_value_heads, head_dim)
            grad_value_states = grad_value_states.view(seq_len, batch, num_key_value_heads, head_dim)

            return grad_query_states, grad_key_states, grad_value_states, None, None, None, None, None

class MegatronQwen2AttentionQKV(torch.nn.Module):
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
        return MegatronQwen2AttentionQKVFunc.apply(
            query_states,
            key_states,
            value_states,
            cos,
            sin,
            attention_mask,
            softmax_scale,
            attention_dropout)

def qwen2_attn_qkv_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None, # no support
    ):

    assert packed_seq_params is None, "packed sequence is not supported in Qwen2AttentionQKV"

    if(not hasattr(self, 'attn_qkv')):
        self.attn_qkv = MegatronQwen2AttentionQKV()

    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states) # (seq, batch, num_heads, head_dim)
    key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        inference_params, key, value, rotary_pos_emb
    )

    if rotary_pos_emb is not None:
        cos, sin = torch.cos(rotary_pos_emb), torch.sin(rotary_pos_emb) #　(seq, 1, 1, head_dim), fp32

    attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype, device=query.device).masked_fill_(attention_mask, -10000.0)

    seq_len, batch, num_heads, head_dim = query.size()
    softmax_scale = 1.0 / head_dim ** 0.5

    core_attn_out = self.attn_qkv(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        cos,
        sin,
        attention_mask,
        softmax_scale,
        self.config.hidden_dropout)

    core_attn_out = core_attn_out.view(seq_len, batch, -1)

    attn_output, bias = self.linear_proj(core_attn_out)

    return attn_output, bias

def fuse_megatron_qwen2_attn_qkv():
    import megatron_patch
    from megatron_patch.model.qwen2.transformer.attention import Attention
    megatron_patch.model.qwen2.transformer.attention.Attention.forward = qwen2_attn_qkv_forward        