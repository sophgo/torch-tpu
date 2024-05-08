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
		Tensor &output);

	Tensor a16_matmul_forward(
		Tensor &active,
		Tensor &weight,
		Tensor &scale,
		Tensor &zp,
        int8_t group_size,
		int8_t weight_bits,
		Tensor &output);

	Tensor a16_llama_mlp_forward(
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

}