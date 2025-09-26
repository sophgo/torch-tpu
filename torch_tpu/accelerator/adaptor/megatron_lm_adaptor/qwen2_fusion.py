import torch
from megatron_lm_adaptor.megatron_fusion.attention import MegatronQwen2AttentionQKV
from megatron_lm_adaptor.megatron_fusion.mlp import MegatronQwen2MlpFunc
from torch_tpu.tpu.custom_op.rmsnorm import llama_rmsnorm_forward

########################################################
# Attention
########################################################
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

    query_states = query.contiguous()
    if rotary_pos_emb is not None:
      seq_len = rotary_pos_emb.shape[0]
      if seq_len != self._rope_cached_seq_len:
        self._rope_cos_cached, self._rope_sin_cached = torch.cos(rotary_pos_emb), torch.sin(rotary_pos_emb) #　(seq, 1, 1, head_dim), fp32
        self._rope_cos_cached = self._rope_cos_cached.to(query_states.dtype)
        self._rope_sin_cached = self._rope_sin_cached.to(query_states.dtype)
        self._rope_cached_seq_len = seq_len

    attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype, device=query.device).masked_fill_(attention_mask, -10000.0)


    seq_len, batch, num_heads, head_dim = query.size()  #8 512 14 128
    softmax_scale = 1.0 / head_dim ** 0.5
    #assign buffers for attn_output, softmax_lse
    max_atten_fwd = seq_len * batch * num_heads * head_dim
    max_atten_softmax = seq_len * batch * num_heads
    if self.atten_fwd_atten_output.numel() < max_atten_fwd:
      self.atten_fwd_atten_output = torch.empty(
                    max_atten_fwd, dtype=query_states.dtype, device=query_states.device)
    if self.atten_fwd_softmax_lse.numel() < max_atten_softmax:
      self.atten_fwd_softmax_lse = torch.empty(
                    max_atten_softmax, dtype=torch.float32, device=query_states.device)

    core_attn_out = self.attn_qkv(
        query_states,
        key.contiguous(),
        value.contiguous(),
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

def fuse_megatron_qwen2_attn_qkv():
    import megatron_patch
    from megatron_patch.model.qwen2.transformer.attention import Attention
    ori_init = Attention.__init__
    #assign buffers for cos, sin and atten_fwd_output and softmax
    def patched_init(self, *args, **kwargs):
        ori_init(self, *args, **kwargs)
        self.register_buffer("_rope_cos_cached", None, persistent=False)
        self.register_buffer("_rope_sin_cached", None, persistent=False)
        self.register_buffer("atten_fwd_atten_output", torch.empty(1), persistent=False)
        self.register_buffer("atten_fwd_softmax_lse", torch.empty(1), persistent=False)
        self._rope_cached_seq_len = 0

    Attention.__init__ = patched_init
    megatron_patch.model.qwen2.transformer.attention.Attention.forward = qwen2_attn_qkv_forward


########################################################
# MLP
########################################################
def fuse_megatron_qwen2_mlp():
    import megatron
    from megatron.core.parallel_state import get_tensor_model_parallel_group
    def hook(grad):
        if get_tensor_model_parallel_group():
            torch.distributed.all_reduce(grad, group=get_tensor_model_parallel_group())
        return grad
    def qwen2_tpu_mlp_forward(self, hidden_states):
        if self.config.bias_activation_fusion and self.activation_func == F.silu and self.config.gated_linear_unit:
            # note: megatron qwen2 hidden_states shape=(seq_len, batch, hidden_size), different from transformers llama_mlp (batch, seq_len, hidden_size)
            b, s, h = hidden_states.shape #s for seq_len, b for batch, h for hidden_size
            i = self.linear_fc2.weight.shape[1]  # hidden_size // 2
            max_fwd = s*b * (2 * i + h) #for three forward buffers
            max_bwd = s*b * (h + i) + 3 * i * h #for five backward buffers
            #if the bufffer is less than the max_fwd or max_bwd, we need to resize it
            if self.mlp_scratch_fwd.numel() < max_fwd:
                self.mlp_scratch_fwd = torch.empty(
                    max_fwd, dtype=self.linear_fc1.weight.dtype, device=self.linear_fc1.weight.device
                )

            if self.mlp_scratch_bwd.numel() < max_bwd:
                self.mlp_scratch_bwd = torch.empty(
                    max_bwd, dtype=self.linear_fc1.weight.dtype, device=self.linear_fc1.weight.device
                )
            buf_fwd = self.mlp_scratch_fwd
            buf_bwd = self.mlp_scratch_bwd
            output = MegatronQwen2MlpFunc.apply(
                hidden_states,
                self.linear_fc1.weight,
                self.linear_fc2.weight,
                buf_fwd,
                buf_bwd
            )
            if get_tensor_model_parallel_group():
                torch.distributed.all_reduce(output, group=get_tensor_model_parallel_group())
            if hidden_states.requires_grad:
                hidden_states.register_hook(hook)
            return output, None
        else:
            ValueError("MegatronDeepSpeedTPU only support silu activation func")
    import megatron_patch
    from megatron_patch.model.qwen2.transformer.mlp import MLP
    _orig_mlp_init = MLP.__init__
    def _mlp_init_with_buf(self, *args, **kwargs):
        _orig_mlp_init(self, *args, **kwargs)
        device = self.linear_fc1.weight.device
        dtype = self.linear_fc1.weight.dtype
        #preset a minimum buffer for forward and backward
        self.register_buffer('mlp_scratch_fwd',
           torch.empty(1, dtype=dtype, device=device),
            persistent=False
        )
        self.register_buffer('mlp_scratch_bwd',
            torch.empty(1, dtype=dtype, device=device),
            persistent=False
        )
    MLP.__init__ = _mlp_init_with_buf
    megatron_patch.model.qwen2.transformer.mlp.MLP.forward = qwen2_tpu_mlp_forward


########################################################
# RMSNorm
########################################################
def fuse_megatron_qwen2_rmsnorm():
    import megatron_patch
    from megatron_patch.model.qwen2.rms_norm import Qwen2RMSNorm
    megatron_patch.model.qwen2.rms_norm.Qwen2RMSNorm.forward = llama_rmsnorm_forward