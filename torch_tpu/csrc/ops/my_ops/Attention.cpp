#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
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
		Tensor &out)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w_attn);
		CHECK_TENSOR_IN_DEVICE(w_proj);
		if (b_attn.has_value())
			CHECK_TENSOR_IN_DEVICE(b_attn.value());
		if (b_proj.has_value())
			CHECK_TENSOR_IN_DEVICE(b_proj.value());
		CHECK_TENSOR_IN_DEVICE(q);
		CHECK_TENSOR_IN_DEVICE(k);
		CHECK_TENSOR_IN_DEVICE(v);
		CHECK_TENSOR_IN_DEVICE(softmax_out);
		CHECK_TENSOR_IN_DEVICE(soft_v);
		CHECK_TENSOR_IN_DEVICE(out);

		TIMING_START;
        TORCH_CHECK(0, "Not implemented");
		TIMING_END(tpu::ATTN_FORWARD);
		return out;
	}

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
		const c10::optional<Tensor> &grad_b_proj)
	{
		CHECK_TENSOR_IN_DEVICE(grad_output);
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w_attn);
		CHECK_TENSOR_IN_DEVICE(w_proj);
		CHECK_TENSOR_IN_DEVICE(q);
		CHECK_TENSOR_IN_DEVICE(k);
		CHECK_TENSOR_IN_DEVICE(v);
		CHECK_TENSOR_IN_DEVICE(softmax_out);
		CHECK_TENSOR_IN_DEVICE(soft_v);
		CHECK_TENSOR_IN_DEVICE(bias);
		CHECK_TENSOR_IN_DEVICE(grad_input);
		CHECK_TENSOR_IN_DEVICE(grad_w_attn);
		CHECK_TENSOR_IN_DEVICE(grad_w_proj);
		if (grad_b_attn.has_value())
			CHECK_TENSOR_IN_DEVICE(grad_b_attn.value());
		if (grad_b_proj.has_value())
			CHECK_TENSOR_IN_DEVICE(grad_b_proj.value());

		TIMING_START;
        TORCH_CHECK(0, "Not implemented");
		TIMING_END(tpu::ATTN_BACKWARD);
		return std::tuple<Tensor, Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>>(grad_input, grad_w_attn, grad_w_proj, grad_b_attn, grad_b_proj);
	}
}
