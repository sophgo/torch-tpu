#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
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
		TIMING_START;
#if defined BACKEND_SG2260
		tpuRtStatus_t status = sgdnnRMSNorm(
			c10_tpu::getCurrentTPUStream(),
			tpu::TPUGenerateSgdnnTensor(input),
			scale.has_value() ? tpu::TPUGenerateSgdnnTensor(scale.value()) : sgdnnUndefinedTensor(),
			bias.has_value() ? tpu::TPUGenerateSgdnnTensor(bias.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(output),
			axis,
			eps,
			1.,
			scale.has_value(),
			bias.has_value());
		TORCH_CHECK(status == tpuRtSuccess);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END(tpu::RMSNORM_FORWARD);
		return output;
	}

}
