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
		bm_status_t status = sgdnnLLamaMlp(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(weight0),
			tpu::TPUGenerateSgdnnTensor(weight1),
			tpu::TPUGenerateSgdnnTensor(weight2),
			tpu::TPUGenerateSgdnnTensor(output));
		TORCH_CHECK(status == BM_SUCCESS);
		TIMING_END(tpu::LLAMA_MLP_FORWARD);
		return output;
	}

}
