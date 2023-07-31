#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

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
		Tensor &out2)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w1);
		CHECK_TENSOR_IN_DEVICE(w2);
		if (b1.has_value())
			CHECK_TENSOR_IN_DEVICE(b1.value());
		if (b2.has_value())
			CHECK_TENSOR_IN_DEVICE(b2.value());
		CHECK_TENSOR_IN_DEVICE(out1);
		CHECK_TENSOR_IN_DEVICE(p);
		CHECK_TENSOR_IN_DEVICE(out2);
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
		bm_status_t status = sgdnnMlp(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(w1),
			tpu::TPUGenerateSgdnnTensor(w2),
			b1.has_value() ? tpu::TPUGenerateSgdnnTensor(b1.value()) : sgdnnUndefinedTensor(),
			b2.has_value() ? tpu::TPUGenerateSgdnnTensor(b2.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(out1),
			tpu::TPUGenerateSgdnnTensor(p),
			tpu::TPUGenerateSgdnnTensor(out2));
		TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::MLP_FORWARD, timer.ElapsedUS());
#endif

		return out2;
	}

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
		const c10::optional<Tensor> &grad_b2)
	{
		CHECK_TENSOR_IN_DEVICE(grad_output);
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w1);
		CHECK_TENSOR_IN_DEVICE(w2);
		CHECK_TENSOR_IN_DEVICE(out1);
		CHECK_TENSOR_IN_DEVICE(p);
		CHECK_TENSOR_IN_DEVICE(grad_input);
		CHECK_TENSOR_IN_DEVICE(grad_w1);
		CHECK_TENSOR_IN_DEVICE(grad_w2);
		if (grad_b1.has_value())
			CHECK_TENSOR_IN_DEVICE(grad_b1.value());
		if (grad_b2.has_value())
			CHECK_TENSOR_IN_DEVICE(grad_b2.value());
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
		bm_status_t status = sgdnnMlpBackward(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(grad_output),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(w1),
			tpu::TPUGenerateSgdnnTensor(w2),
			tpu::TPUGenerateSgdnnTensor(out1),
			tpu::TPUGenerateSgdnnTensor(p),
			tpu::TPUGenerateSgdnnTensor(grad_input),
			tpu::TPUGenerateSgdnnTensor(grad_w1),
			tpu::TPUGenerateSgdnnTensor(grad_w2),
			grad_b1.has_value() ? tpu::TPUGenerateSgdnnTensor(grad_b1.value()) : sgdnnUndefinedTensor(),
			grad_b2.has_value() ? tpu::TPUGenerateSgdnnTensor(grad_b2.value()) : sgdnnUndefinedTensor());
		TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::MLP_BACKWARD, timer.ElapsedUS());
#endif
		return std::tuple<Tensor, Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>>(grad_input, grad_w1, grad_w2, grad_b1, grad_b2);
	}
}
