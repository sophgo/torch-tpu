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
	Tensor a16_matmul_forward(
		Tensor &active,
		Tensor &weight,
		Tensor &scale,
		Tensor &zp,
        int8_t group_size,
		int8_t weight_bits,
		Tensor &output)
	{
		CHECK_TENSOR_IN_DEVICE(active);
		CHECK_TENSOR_IN_DEVICE(weight);
		CHECK_TENSOR_IN_DEVICE(scale);
		CHECK_TENSOR_IN_DEVICE(zp);
		CHECK_TENSOR_IN_DEVICE(output);

		TIMING_START;
		tpu_status_t status = sgdnnLLamaA16Matmul(
			tpu::TPUGetDeviceResource(),
			tpu::TPUGenerateSgdnnTensor(active),
			tpu::TPUGenerateSgdnnTensor(weight),
			tpu::TPUGenerateSgdnnTensor(scale),
			tpu::TPUGenerateSgdnnTensor(zp),
            group_size,
			weight_bits,
			tpu::TPUGenerateSgdnnTensor(output));
		TORCH_CHECK(status == SG_SUCCESS);
		TIMING_END(tpu::LLama2A16MATMUL);
		return output;
	}

}
