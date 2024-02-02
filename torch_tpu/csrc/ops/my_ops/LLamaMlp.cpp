#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor llama_mlp_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &weight1,
		Tensor &weight2,
		Tensor &output)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(weight0);
		CHECK_TENSOR_IN_DEVICE(weight1);
		CHECK_TENSOR_IN_DEVICE(weight2);
		CHECK_TENSOR_IN_DEVICE(output);

		TIMING_START;
#if defined BACKEND_SG2260
		tpuRtStatus_t status = sgdnnLLamaMlp(
			c10_tpu::getCurrentTPUStream(),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(weight0),
			tpu::TPUGenerateSgdnnTensor(weight1),
			tpu::TPUGenerateSgdnnTensor(weight2),
			tpu::TPUGenerateSgdnnTensor(output));
		TORCH_CHECK(status == tpuRtSuccess);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END(tpu::LLAMA_MLP_FORWARD);
		return output;
	}

}
