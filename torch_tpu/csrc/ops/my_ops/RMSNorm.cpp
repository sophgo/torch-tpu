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
	Tensor rmsnorm_forward(
		Tensor &input,
		const c10::optional<Tensor> &scale,
		const c10::optional<Tensor> &bias,
		Tensor &output,
		int64_t axis,
		double_t eps)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(output);
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
		bm_status_t status = sgdnnRMSNorm(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(input),
			scale.has_value() ? tpu::TPUGenerateSgdnnTensor(scale.value()) : sgdnnUndefinedTensor(),
			bias.has_value() ? tpu::TPUGenerateSgdnnTensor(bias.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(output),
			axis,
			eps,
			1.,
			scale.has_value(),
			bias.has_value());
		TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::RMSNORM_FORWARD, timer.ElapsedUS());
#endif
		return output;
	}

}
