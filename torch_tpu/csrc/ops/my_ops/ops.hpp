#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

namespace at
{
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

	std::tuple<Tensor, Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>> mlp_backward(
		Tensor &grad_output,
		Tensor &input,
		Tensor &w1,
		Tensor &w2,
		Tensor &out1,
		Tensor &p,
		Tensor &grad_input,
		Tensor &grad_w1,
		Tensor &grad_w2,
		const c10::optional<Tensor> &grad_b1,
		const c10::optional<Tensor> &grad_b2);

	Tensor llama_mlp_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &weight1,
		Tensor &weight2,
		const c10::optional<Tensor> &silu,
		const c10::optional<Tensor> &sigmoid,
		const c10::optional<Tensor> &m0,
		Tensor &output,
		bool save_mid_res);

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
		const c10::optional<Tensor> &fetch_slots,
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
		const c10::optional<Tensor> &fetch_slots,
		const c10::optional<Tensor> &mask,
		int64_t	slots_size,
		int64_t mask_size,
		int64_t block_size,
		double C,
		int64_t attention_mode);

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
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &input_lengths, 
        int64_t head,
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
            Tensor &sin, const c10::optional<Tensor> &fetch_slots,
            Tensor &save_slots, const c10::optional<Tensor> &mask, // decode: None
            const Tensor &input_lengths, int64_t head, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t slots_size, int64_t paged_cache_block_size, double C,
            int64_t attention_mode);

        Tensor latent_attention_fp8(
            Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
            Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
            Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
            const c10::optional<Tensor> &mask, // decode: None
            const Tensor &seqlen, int64_t num_heads, int64_t q_lora_rank,
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
            const c10::optional<Tensor> &fetch_slots, Tensor &save_slots,
            const c10::optional<Tensor> &mask, // decode: None
            const Tensor &seqlen, int64_t num_heads, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t quant_block_size, int64_t slots_size,
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
        Tensor &gate_scales,
        Tensor &up_scales,
        Tensor &down_scales,
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
}
