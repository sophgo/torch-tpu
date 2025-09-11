#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

namespace at
{

/**
 * @brief Gated MLP (Multi-Layer Perceptron) with SwiGLU activation function
 *
 * Implements a gated feed-forward network commonly used in transformer architectures:
 * FFN(x) = down_proj(SwiGLU(gate_proj(x), up_proj(x)))
 * where SwiGLU(a, b) = silu(a) * b
 *
 * @param input Input tensor of shape [..., hidden_size]
 * @param up_proj Up projection weight matrix of shape [intermediate_size, hidden_size]
 * @param gate_proj Gate projection weight matrix of shape [intermediate_size, hidden_size]
 * @param down_proj Down projection weight matrix of shape [intermediate_size, hidden_size]
 * @param up_bias Optional bias for up projection of shape [intermediate_size]
 * @param gate_bias Optional bias for gate projection of shape [intermediate_size]
 * @param down_bias Optional bias for down projection of shape [hidden_size]
 * @param output Output tensor of same shape as input [..., hidden_size]
 * @return Tensor Reference to the output tensor
 *
 * @details
 * This function implements a gated MLP which uses a Gated Linear Unit (GLU) variant
 * with SiLU activation (SwiGLU). The operations performed are:
 * 1. Projects input through gate_proj and up_proj linear transformations
 * 2. Applies SwiGLU activation: silu(gate_output) * up_output
 * 3. Projects the result through down_proj to match the original dimension
 *
 * This gated architecture is used in Llama, Llama2, and other related models for
 * improved performance over standard MLPs.
 *
 * @note The function performs in-place operations on the output tensor
 * @note This is a specialized MLP variant used in transformer architectures like Llama, Qwen, etc.
 * @see mlp() for standard 2-layer MLP implementation
 */
Tensor gated_mlp(
    const Tensor &input,
    const Tensor& up_proj,
    const Tensor& gate_proj,
    const Tensor& down_proj,
    const c10::optional<Tensor> &up_bias,
    const c10::optional<Tensor> &gate_bias,
    const c10::optional<Tensor> &down_bias,
    Tensor &output);
/**
 * @brief Standard 2-layer MLP (Multi-Layer Perceptron) forward pass
 *
 * Implements a traditional feed-forward network with two linear transformations:
 * MLP(x) = weight2(activation(weight1(x) + bias1) + bias2)
 *
 * @param input Input tensor of shape [..., input_size]
 * @param weight1 First weight matrix of shape [input_size, hidden_size]
 * @param weight2 Second weight matrix of shape [hidden_size, output_size]
 * @param bias1 Optional bias for first linear layer of shape [hidden_size]
 * @param bias2 Optional bias for second linear layer of shape [output_size]
 * @param activation Activation function type ("gelu", "none")
 * @param out Output tensor of shape [..., output_size]
 * @return Tensor Reference to the output tensor
 *
 * @details
 * This function implements a standard MLP with configurable activation function:
 * 1. Linear transformation: weight1 * input + bias1
 * 2. Apply specified activation function
 * 3. Linear transformation: weight2 * activated_result + bias2
 *
 * Supported activation functions:
 * - "gelu": Gaussian Error Linear Unit
 * - "none": No activation (identity)
 *
 * @note This is a simpler MLP variant compared to the gated versions like gated_mlp
 * @see gated_mlp for gated MLP implementation
 */
Tensor mlp(
    const Tensor &input,
    const Tensor &weight1,
    const Tensor &weight2,
    const c10::optional<Tensor> &bias1,
    const c10::optional<Tensor> &bias2,
    const std::string& activation,
    Tensor &out);

/**
 * @brief Standard multi-head attention mechanism
 *
 * Implements the scaled dot-product attention operation commonly used in transformer architectures:
 * Attention(Q, K, V) = softmax(QK^T * scale + mask) * V
 *
 * @param output Output tensor of shape [..., seq_len, num_heads, head_dim]
 * @param query Query tensor of shape [..., seq_len, num_heads, head_dim]
 * @param key Key tensor of shape [..., seq_len, num_heads, head_dim]
 * @param value Value tensor of shape [..., seq_len, num_heads, head_dim]
 * @param pos_cos Optional cosine positional embeddings for rotary positional encoding of shape [..., seq_len, 1, head_dim]
 * @param pos_sin Optional sine positional embeddings for rotary positional encoding of shape [..., seq_len, 1, head_dim]
 * @param mask Optional attention mask to prevent attention to certain positions of shape [seq_len, seq_len]
 * @param softmax_scale Scaling factor applied to the dot product before softmax, typically 1/sqrt(head_dim)
 * @return Tensor The result of attention operation with the same shape as query
 *
 * @details
 * This function computes the standard attention mechanism used in transformer models.
 * It supports rotary positional embeddings (RoPE) through pos_cos and pos_sin parameters,
 * and optional masking for causal or other attention patterns.
 *
 * The attention computation follows these steps:
 * 1. Compute scaled dot-product between query and key tensors
 * 2. Apply rotary positional embeddings if provided
 * 3. Apply mask if provided
 * 4. Apply softmax to get attention weights
 * 5. Multiply attention weights with value tensor
 *
 * @note
 * - This is a core attention implementation used in various transformer-based models.
 * - **And the query, key and value may be packed into a single tensor which is stored in `query`,
 * with shape of [..., seq_len, query_head + key_head + value_head, head_dim]**.
 * - If the query is not contiguous, it means that query, key and value will be performed on the packed tensor.
 * - Usage:
 * - qkv = torch.linear(input, qkv_weight, bias=qkv_bias), # [..., seq_len, query_head + key_head + value_head, head_dim]
 * - q, k, v = torch.split(qkv, [query_head, key_head, value_head], dim=-2) # [..., seq_len, query_head, head_dim], [..., seq_len, key_head, head_dim], [..., seq_len, value_head, head_dim]
 * - and now the attention is performed on packed tensor.
 */
Tensor attention(
    Tensor &output,
    const Tensor &query,
    const Tensor &key,
    const Tensor &value,
    const c10::optional<Tensor> &pos_cos,
    const c10::optional<Tensor> &pos_sin,
    const c10::optional<Tensor> &mask,
    double softmax_scale);

/**
 * @brief Paged attention mechanism V2 implementation for efficient attention computation.
 * Which is optimized with flash attention algorithm.
 *
 * Implements an optimized paged attention operation that utilizes cached key and value tensors
 * with paged memory management. This version supports rotary positional embeddings and 
 * various attention masking strategies.
 * 
 * **The function uniform the prefill and decode stages, and also supports prefill-chunking and prefix-caching,
 * with different input lengths and cache lengths.**
 * 
 * If the value of input lengths is 1, the decode stage is used, otherwise, the prefill stage is used.
 * 
 * **when a request contains both decode and prefill stages,
 * the data and parameters are stored in the same tensor with decode data placed first, followed by prefill data.**
 * 
 * Attention(Q, K, V) = softmax(QK^T * scale + mask) * V
 *
 * @param output Output tensor to store the result
 * @param query Query tensor of shape [..., num_heads, head_dim]
 * @param key Key tensor of shape [..., num_heads, head_dim]
 * @param value Value tensor of shape [..., num_heads, head_dim]
 * @param kcache Cached key tensor for paged attention of shape [..., block_size, num_kv_heads, head_dim]
 * @param vcache Cached value tensor for paged attention of shape [..., block_size, num_kv_heads, head_dim]
 * @param pos_cos Optional cosine positional embeddings for rotary positional encoding of shape [..., seq_len, head_dim]
 * @param pos_sin Optional sine positional embeddings for rotary positional encoding of shape [..., seq_len, head_dim]
 * @param input_lengths Tensor containing the lengths of input sequences of shape [batch]
 * @param cache_lengths Tensor containing the lengths of cached sequences of shape [batch]
 * @param save_slots Tensor indicating where to save the attention results in cache of shape [input_lengths]
 * @param block_tables Tensor containing block tables for paged attention of shape [batch, num_blocks]
 * @param mask Optional attention mask to prevent attention to the future positions. The mask should be of shape [max_seqlen, max_seqlen],
 *             where max_seqlen is the maximum sequence length of the input sequences. The mask should be symmetric and causal.
 *             The mask is only used for the prefill stage. If the mask is None, the function will generate the mask based on the input lengths.
 * @param softmax_scale Scaling factor applied to the dot product before softmax
 * @return Tensor The result of the paged attention operation
 *
 * @details
 * This function computes attention with paged memory management, which is especially useful
 * for autoregressive decoding in large language models. It supports:
 * 1. Rotary positional embeddings through pos_cos and pos_sin parameters
 * 2. Paged key-value caching via Kcache and Vcache
 * 3. Flexible attention masking
 * 4. Configurable softmax scaling
 *
 * The paged attention approach helps reduce memory fragmentation and enables efficient
 * management of key-value caches during inference.
 *
 * @note
 * - This is an optimized version of attention mechanism for inference scenarios.
 * - **If the query is not contiguous, it means that query, key and value will be performed on the packed tensor. See `attention`**
 */
Tensor paged_attention_v2(
    Tensor &output,
    const Tensor &query,
    const Tensor &key,
    const Tensor &value,
    const Tensor &kcache,
    const Tensor &vcache,
    const c10::optional<Tensor> &pos_cos,
    const c10::optional<Tensor> &pos_sin,
    const Tensor &input_lengths,
    const Tensor &cache_lengths,
    const Tensor &save_slots,
    const Tensor &block_tables,
    const c10::optional<Tensor> &mask,
    double softmax_scale);

  /// @cond INTERNAL
	Tensor mlp_forward(
		Tensor &input,
		Tensor &w1,
		Tensor &w2,
		const c10::optional<Tensor> &b1,
		const c10::optional<Tensor> &b2,
		Tensor &out1,
		Tensor &p,
		Tensor &out2);

	Tensor llava_mlp(
		Tensor &input,
		Tensor &w1,
		Tensor &w2,
		const c10::optional<Tensor> &b1,
		const c10::optional<Tensor> &b2,
		Tensor &out);

	Tensor llama_mlp_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &weight1,
		Tensor &weight2,
		const c10::optional<Tensor> &bias0,
		const c10::optional<Tensor> &bias1,
		const c10::optional<Tensor> &bias2,
		const c10::optional<Tensor> &fc1,
		const c10::optional<Tensor> &m0,
		Tensor &output,
    bool save_mid_res);

  std::tuple<Tensor, Tensor, Tensor, Tensor> mlp_backward(
    Tensor &input,
    Tensor &weight0,
    Tensor &weight1,
    Tensor &weight2,
    Tensor& w0x,
    Tensor& w1x,
    Tensor& output,
    Tensor &grad_tmp,
    Tensor &grad_input,
    Tensor &grad_weight0,
    Tensor &grad_weight1,
    Tensor &grad_weight2);


	Tensor matmul_gptq_forward(
		Tensor &active,
		Tensor &weight,
		const c10::optional<Tensor> &bias,
		Tensor &scale,
		Tensor &zp,
        int8_t group_size,
		int8_t weight_bits,
		Tensor &output);

	Tensor mlp_w8a16_dq_forward(
		Tensor &input,
		Tensor &gate_weight,
		Tensor &up_weight,
		Tensor &down_weight,
		Tensor &gate_scale,
		Tensor &up_scale,
		Tensor &down_scale,
		Tensor &output,
		int64_t blocksize);

	Tensor mm_w8a16_dq_forward(
		Tensor &input,
		Tensor &weight,
		Tensor &scale,
		Tensor &output,
		int64_t blocksize);

	Tensor mlp_w8a8_quant_forward(
		Tensor &input,
		Tensor &gate_weight,
		Tensor &up_weight,
		Tensor &down_weight,
		Tensor &gate_scale,
		Tensor &up_scale,
		Tensor &down_scale,
		Tensor &output,
		int64_t blocksize);

	Tensor llama_mlp_gptq_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &zp0,
		Tensor &sale0,
		Tensor &weight1,
		Tensor &zp1,
		Tensor &sale1,
		Tensor &weight2,
		Tensor &zp2,
		Tensor &sale2,
		int64_t group_size,
		int64_t weight_bits,
		Tensor &output);

	Tensor rmsnorm_forward(
		Tensor &input,
		const c10::optional<Tensor> &scale,
		const c10::optional<Tensor> &bias,
		Tensor &output,
		int64_t axis,
		double_t eps);

	Tensor rmsnorm_backward(
		Tensor &grad_output,
		Tensor &input,
		const c10::optional<Tensor> &scale,
		const c10::optional<Tensor> &bias,
		Tensor &rms,
		const c10::optional<Tensor> &grad_input,
		const c10::optional<Tensor> &grad_scale,
		const c10::optional<Tensor> &grad_bias,
		int64_t axis,
		double_t eps);

	Tensor paged_attention(
		Tensor &OUT,
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &Kcache,
		Tensor &Vcache,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const Tensor &input_lengths,
		const c10::optional<Tensor> &cache_lengths,
		const Tensor &save_slots,
		const c10::optional<Tensor> &block_tables,
		const c10::optional<Tensor> &mask,
        int64_t rope_head_size,
		int64_t	slots_size,
		int64_t	fetch_size,
		int64_t mask_size,
		int64_t block_size,
		double C,
		int64_t attention_mode);

	Tensor llama_attention(
		Tensor &OUT,
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &Kcache,
		Tensor &Vcache,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const Tensor &input_lengths,
		const Tensor &save_slots,
		const c10::optional<Tensor> &block_tables,
		const c10::optional<Tensor> &mask,
		int64_t	slots_size,
		int64_t mask_size,
		int64_t block_size,
		double C,
		int64_t attention_mode);

	Tensor hybrid_attention(
		Tensor &OUT,
		Tensor &mode_tensor,
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &Kcache,
		Tensor &Vcache,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const Tensor &input_lengths,
		const Tensor &cache_lengths,
		const Tensor &prompt_lengths,
		const Tensor &slots,
		const Tensor &block_tables,
		const c10::optional<Tensor> &mask,
		int64_t	slots_size,
		int64_t mask_size,
		int64_t block_size,
		double C);

	Tensor llama_attention_forward(
		Tensor &OUT,
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const c10::optional<Tensor> &mask,
		const c10::optional<Tensor> &softmax_lse,
		const Tensor &input_lengths,
		int64_t mask_size,
		double C,
		double dropout_rate,
		int64_t batch);

	std::tuple<Tensor, Tensor, Tensor> llama_attention_backward(
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &O,
		Tensor &dO,
		Tensor &l,
		Tensor &dQ,
		Tensor &dK,
		Tensor &dV,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const c10::optional<Tensor> &mask,
		const Tensor &input_lengths,
		int64_t mask_max,
		double C
		);

	Tensor attn_forward(
		Tensor &input,
		Tensor &w_attn,
		Tensor &w_proj,
		const c10::optional<Tensor> &b_attn,
		const c10::optional<Tensor> &b_proj,
		Tensor &q,
		Tensor &k,
		Tensor &v,
		Tensor &softmax_out,
		Tensor &soft_v,
		Tensor &out);

	Tensor llava_attention(
	    Tensor &OUT,
	    Tensor &Q,
	    Tensor &K,
	    Tensor &V,
	    const c10::optional<Tensor> &cos,
	    const c10::optional<Tensor> &sin,
	    const c10::optional<Tensor> &mask,
	    double C);

	Tensor latent_attention(
        Tensor &OUT,
        Tensor &Q,
        Tensor &KV,
        Tensor &PE,
        Tensor &WUQ,
        Tensor &WUKV,
        Tensor &KVcache,
        Tensor &PEcache,
        Tensor &cos,
        Tensor &sin,
		const c10::optional<Tensor> &KVU,
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &input_lengths,
        int64_t head,
        int64_t generate_token,
        int64_t q_lora_rank,
        int64_t kv_lora_rank,
        int64_t qk_nope_head_dim,
        int64_t qk_rope_head_dim,
        int64_t v_head_dim,
        int64_t mask_size,
        int64_t max_cache_size,
        double C,
        int64_t attention_mode);

        Tensor paged_latent_attention(
            Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
            Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
            Tensor &sin, const c10::optional<Tensor> &block_table,
            Tensor &save_slots, const c10::optional<Tensor> &KVU, const c10::optional<Tensor> &mask, // decode: None
            const Tensor &input_lengths, const Tensor &cache_lengths,
			int64_t head, int64_t generate_token, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t max_paged_block_num, int64_t paged_cache_block_size, double C,
            int64_t attention_mode);

        Tensor latent_attention_fp8(
            Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
            Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
            Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
            const c10::optional<Tensor> &KVU, const c10::optional<Tensor> &mask, // decode: None
            const Tensor &seqlen, int64_t num_heads, int64_t generate_token, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t quant_block_size, int64_t max_cache_size,
            double softmax_scale,
            int64_t attention_mode // prefille 0, decode 1
        );

        Tensor paged_latent_attention_fp8(
            Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
            Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
            Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
            const c10::optional<Tensor> &block_table, Tensor &save_slots,
			const c10::optional<Tensor> &KVU, const c10::optional<Tensor> &mask, // decode: None
            const Tensor &seqlen, const Tensor &cache_seqlen,
			int64_t num_heads, int64_t generate_token, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t quant_block_size, int64_t max_paged_block_num,
            int64_t paged_cache_block_size, double softmax_scale,
            int64_t attention_mode);

        std::tuple<Tensor, Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>> attn_backward(
		Tensor &grad_output,
		Tensor &input,
		Tensor &w_attn,
		Tensor &w_proj,
		Tensor &q,
		Tensor &k,
		Tensor &v,
		Tensor &softmax_out,
		Tensor &soft_v,
		Tensor &bias,
		Tensor &grad_input,
		Tensor &grad_w_attn,
		Tensor &grad_w_proj,
		const c10::optional<Tensor> &grad_b_attn,
		const c10::optional<Tensor> &grad_b_proj);

	std::tuple<Tensor, Tensor, Tensor> ln_mm_forward(
		Tensor &input,
        Tensor &w,
		const c10::optional<Tensor> &b,
		Tensor &gamma,
		Tensor &beta,
        double eps,
        Tensor &mean,
        Tensor &rstd,
		Tensor &out);

	std::tuple<Tensor, Tensor, Tensor> ln_mm_backward(
        Tensor &grad_out_ln,
        Tensor &input,
		Tensor &mean,
		Tensor &rstd,
		Tensor &gamma,
		Tensor &grad_input,
        Tensor &grad_gamma,
        Tensor &grad_beta);

	std::tuple<Tensor, Tensor, Tensor, Tensor> add_ln_mm_forward(
        Tensor &input0,
        Tensor &input1,
        Tensor &w,
		const c10::optional<Tensor> &b,
		Tensor &gamma,
		Tensor &beta,
        double eps,
        Tensor &out_add,
        Tensor &mean,
        Tensor &rstd,
		Tensor &out);

	std::tuple<Tensor, Tensor, Tensor> add_ln_mm_backward(
        Tensor &grad_out_ln,
        Tensor &input,
		Tensor &mean,
		Tensor &rstd,
		Tensor &gamma,
		Tensor &grad_input,
        Tensor &grad_gamma,
        Tensor &grad_beta);

	Tensor lora_matmul_forward(
        Tensor &input,
        Tensor &loraA,
        Tensor &loraB,
        Tensor &weight,
        Tensor &output,
        double_t scale);

    Tensor fused_moe_grouped_topk(
        Tensor &topk_experts_res,
        Tensor &topk_weights_res_bf16,
        Tensor &left_bf16,
        Tensor &right_bf16,
        Tensor &topk_weights_res,
        Tensor &left,
        Tensor &right,
        Tensor &max,
        Tensor &matmul_res,
        Tensor &softmax_res);

    Tensor fused_moe_fused_experts(
        Tensor &output,
        Tensor &input,
        const c10::optional<Tensor> &output_sample,
        const c10::optional<Tensor> &input_sample,
        Tensor &gate_weights,
        Tensor &up_weights,
        Tensor &down_weights,
        const c10::optional<Tensor>  &gate_scales,
        const c10::optional<Tensor>  &up_scales,
        const c10::optional<Tensor>  &down_scales,
        Tensor &select_experts,
        Tensor &routing_weights,
        const c10::optional<Tensor> &num_select_experts,
        const c10::optional<Tensor> &select_experts_middle,
        const c10::optional<Tensor> &routing_weights_middle,
        int64_t blocksize,
        int64_t num_experts,
        int64_t num_experts_per_tok,
        bool use_grouped_topk,
        int64_t num_expert_group,
        int64_t topk_group,
        const c10::optional<Tensor> &silu,
        const c10::optional<Tensor> &sigmoid,
        const c10::optional<Tensor> &m0,
        bool save_mid_res);

	void TGI_input_ids_update_decode_phase(
		Tensor &all_input_ids,
		Tensor &next_ids,
		IntArrayRef input_lengths,
		int64_t n_accept_ids = 1);

	void enable_pmu();

	void disable_pmu();

	void enable_profile(
		c10::optional<int64_t> max_record_num,
		c10::optional<int64_t> mode);
	void disable_profile();

	void reset_tpudnn_optimer();
	void dump_tpudnn_optimer();

	void dynlib_execute(
		const std::string &so_url,
		const std::string &func_name,
		const std::vector<Tensor> &tensors,
		const std::vector<int64_t> &tensors_index,
		const std::vector<double> &fp_scalars,
		const std::vector<int64_t> &fp_scalars_index,
		const std::vector<int64_t> &fixed_scalars,
        const std::vector<int64_t> &fixed_scalars_index);

	void FormatCast(
		const Tensor &self,
		const Tensor &dst,
		int64_t tpu_format);

	Tensor nms(
		const Tensor& dets,
		const Tensor& scores,
		double iou_threshold);

	Tensor noaux_tc_topk(
		Tensor &values,
		Tensor &indices, 
		const Tensor &scores,
		int64_t n_groups,
		int64_t topk_groups,
		int64_t top_k);
  /// @endcond
}
