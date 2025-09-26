import torch
import torch.nn as nn
import megatron
from megatron import core
from torch import Tensor
from typing import NoReturn, Optional, Tuple, Union
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import Attention


import os
from distutils.util import strtobool
whole_net_trans = strtobool(os.environ.get("QWEN2_WHOLE_NET_TRANS", "0")) or strtobool(os.environ.get("WHOLE_NET_TRANS", "0"))

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
                atten_fwd_atten_output,
                atten_fwd_softmax_lse,
                attention_dropout):
        # cos, sin: (seq, 1, 1, head_dim), fp32
        # attention_mask: (batch, 1, seq, seq), bool
        ctx.softmax_scale = softmax_scale
        if whole_net_trans:
            # q, k, v: (seq, batch, num_heads, head_dim)
            batch, seq_len, num_attn_head, head_dim = query_states.size()
            output_shape = (batch, seq_len, num_attn_head, head_dim)
            cos = cos.view(seq_len, 1, head_dim)
            sin = sin.view(seq_len, 1, head_dim)
        else:
            # q, k, v: (batch, seq, num_heads, head_dim)
            seq_len, batch, num_attn_head, head_dim = query_states.size()
            output_shape = (seq_len, batch, num_attn_head, head_dim)
            cos = cos.to(query_states.dtype).view(seq_len, 1, head_dim)
            sin = sin.to(query_states.dtype).view(seq_len, 1, head_dim)
            query_states = query_states.transpose(0, 1)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)
        num_kv_head = key_states.size(-2)

        query_states = query_states.view(batch * seq_len, num_attn_head, head_dim)
        key_states = key_states.view(batch * seq_len, num_kv_head, head_dim)
        value_states = value_states.view(batch * seq_len, num_kv_head, head_dim).contiguous()

        if attention_mask is not None:
            if whole_net_trans:
                if attention_mask.numel() > seq_len * seq_len:
                    attention_mask_2d = attention_mask.view(batch, seq_len, seq_len)
                else:
                    attention_mask_2d = attention_mask.view(seq_len, seq_len).unsqueeze(0).repeat(batch, 1, 1).view(batch * seq_len, seq_len)
            else:
                if attention_mask.numel() > seq_len * seq_len:
                    attention_mask_2d = attention_mask.view(batch, seq_len, seq_len)
                else:
                    attention_mask_2d = attention_mask.view(seq_len, seq_len).unsqueeze(0).repeat(batch, 1, 1).view(batch * seq_len, seq_len)
        else:
            attention_mask_2d = None

        attn_output = atten_fwd_atten_output.view_as(query_states)
        softmax_lse = atten_fwd_softmax_lse.view([batch * seq_len, num_attn_head, 1])
        input_length = torch.tensor([seq_len] * batch, dtype=torch.int32)#TODO:construct data in device
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
            cos = cos.view(seq_len, 1, head_dim)
            sin = sin.view(seq_len, 1, head_dim)

            grad_query_states = torch.empty_like(query_states)
            grad_key_states = torch.empty_like(key_states)
            grad_value_states = torch.empty_like(value_states)

            if attention_mask is not None:
                if attention_mask.numel() > seq_len * seq_len:
                    attention_mask = attention_mask.view(batch * seq_len, seq_len)
                else:
                    attention_mask = attention_mask.view(seq_len, seq_len).unsqueeze(0).repeat(batch, 1, 1).view(batch * seq_len, seq_len)
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

            return grad_query_states, grad_key_states, grad_value_states, None, None, None, None, None, None, None
        else:
            seq_len, batch, num_heads, head_dim = attn_output.size()
            assert seq_len * batch <= 4096, "batch size too large"
            num_key_value_heads = key_states.shape[-2]

            grad_query_states = torch.empty_like(query_states)
            grad_key_states = torch.empty_like(key_states)
            grad_value_states = torch.empty_like(value_states)

            query_states = query_states.view(batch * seq_len, num_heads, head_dim)
            key_states = key_states.view(batch * seq_len, num_key_value_heads, head_dim)
            value_states = value_states.view(batch * seq_len, num_key_value_heads, head_dim)


            attn_output = attn_output.view(batch * seq_len, num_heads, head_dim)
            grad_output = grad_output.view(batch * seq_len, num_heads, head_dim)
            lse = lse.view(batch * seq_len, num_heads, 1)
            cos = cos.view(seq_len, 1, head_dim)
            sin = sin.view(seq_len, 1, head_dim)

            grad_query_states = grad_query_states.view(batch * seq_len, num_heads, head_dim)
            grad_key_states = grad_key_states.view(batch * seq_len, num_key_value_heads, head_dim)
            grad_value_states = grad_value_states.view(batch * seq_len, num_key_value_heads, head_dim)

            input_lengths = torch.tensor([seq_len]* batch, dtype=torch.int32, device=query_states.device)

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

            return grad_query_states, grad_key_states, grad_value_states, None, None, None, None, None, None, None

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
                atten_fwd_atten_output,
                atten_fwd_softmax_lse,
                attention_dropout = 0.0):
        return MegatronQwen2AttentionQKVFunc.apply(
            query_states,
            key_states,
            value_states,
            cos,
            sin,
            attention_mask,
            softmax_scale,
            atten_fwd_atten_output,
            atten_fwd_softmax_lse,
            attention_dropout)



def megatron_attn_qkv_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
    hidden_states = hidden_states.contiguous()
    no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
    if no_rope:
            assert False, "Torch_tpu do not support 'no rope'"

    assert packed_seq_params is None, "packed sequence is not supported in Qwen2AttentionQKV"
    # hidden_states: [sq, b, h]
    if self.config.flash_decode and not self.training and inference_context is not None:
        rotary_pos_emb = None
    else:
        assert rotary_pos_cos is None and rotary_pos_sin is None

    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

    in_decode_mode = (
        inference_context is not None
        and inference_context.is_decode_only()
        and not self.training
    )

    query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
        self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
    )
    if(not hasattr(self, 'attn_qkv')):
        self.attn_qkv = MegatronQwen2AttentionQKV()

    query_states = query.contiguous()
    if rotary_pos_emb is not None:
        if rotary_pos_emb.shape[0] != self._rope_cached_seq_len:
            self._rope_cos_cached = torch.cos(rotary_pos_emb).to(query_states.dtype)
            self._rope_sin_cached = torch.sin(rotary_pos_emb).to(query_states.dtype)
            self._rope_cached_seq_len = rotary_pos_emb.shape[0]

    seq_len, batch, num_heads, head_dim = query.size()
    softmax_scale = 1.0 / head_dim ** 0.5
    if self.config.apply_query_key_layer_scaling:
        softmax_scale /= self.layer_number
    #assign buffers for attn_output, softmax_lse
    max_atten_fwd = seq_len * batch * num_heads * head_dim
    max_atten_softmax = seq_len * batch * num_heads

    if not hasattr(self, 'atten_fwd_atten_output') or self.atten_fwd_atten_output.numel() < max_atten_fwd:
        self.atten_fwd_atten_output = torch.empty(
                        max_atten_fwd, dtype=query_states.dtype, device=query_states.device)
    if not hasattr(self, 'atten_fwd_softmax_lse') or self.atten_fwd_softmax_lse.numel() < max_atten_softmax:
        self.atten_fwd_softmax_lse = torch.empty(
                        max_atten_softmax, dtype=torch.float32, device=query_states.device)

    attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype, device=query.device).masked_fill_(attention_mask, -10000.0)
    core_attn_out = self.attn_qkv(
        query_states,
        key,
        value,
        self._rope_cos_cached,
        self._rope_sin_cached,
        attention_mask,
        softmax_scale,
        self.atten_fwd_atten_output,
        self.atten_fwd_softmax_lse,
        self.config.hidden_dropout)
    core_attn_out = core_attn_out.view(seq_len, batch, -1)
    attn_output, bias = self.linear_proj(core_attn_out)
    return attn_output, bias

def fuse_megatron_attn_forward():
    ori_init = Attention.__init__
    #assign buffers for cos, sin and atten_fwd_output and softmax
    def patched_init(self, *args, **kwargs):
        ori_init(self, *args, **kwargs)
        # Initialize empty buffers for attention forward/backward caching
        self.register_buffer("_rope_cos_cached", None, persistent=False)
        self.register_buffer("_rope_sin_cached", None, persistent=False)
        self.register_buffer("atten_fwd_atten_output", torch.empty(1), persistent=False)
        self.register_buffer("atten_fwd_softmax_lse", torch.empty(1), persistent=False)
        self._rope_cached_seq_len = 0

    Attention.__init__ = patched_init
    Attention.forward = megatron_attn_qkv_forward
