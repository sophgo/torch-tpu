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
	Tensor matmul_gptq_forward(
		Tensor &active,
		Tensor &weight,
		const c10::optional<Tensor> &bias,
		Tensor &scale,
		Tensor &zp,
        int8_t group_size,
		int8_t weight_bits,
		Tensor &output)
	{
		CHECK_TENSOR_IN_DEVICE(active);
		CHECK_TENSOR_IN_DEVICE(weight);
		if (bias.has_value())
			CHECK_TENSOR_IN_DEVICE(bias.value());
		CHECK_TENSOR_IN_DEVICE(scale);
		CHECK_TENSOR_IN_DEVICE(zp);
		CHECK_TENSOR_IN_DEVICE(output);

		TIMING_START;
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLLamaA16MatmulAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, active),
			tpu::TPUGenerateTpudnnTensor( stream, weight),
			bias.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, bias.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor( stream, scale),
			tpu::TPUGenerateTpudnnTensor( stream, zp),
			group_size,
			weight_bits,
			tpu::TPUGenerateTpudnnTensor( stream, output)
			);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
		TIMING_END(tpu::LLama2A16MATMUL);
		return output;
	}

}
