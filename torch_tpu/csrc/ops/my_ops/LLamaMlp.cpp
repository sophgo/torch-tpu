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
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLLamaMlpAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, weight0),
			tpu::TPUGenerateTpudnnTensor( stream, weight1),
			tpu::TPUGenerateTpudnnTensor( stream, weight2),
			tpu::TPUGenerateTpudnnTensor( stream, output));
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END(tpu::LLAMA_MLP_FORWARD);
		return output;
	}

	Tensor llama_mlp_gptq_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &zp0,
		Tensor &scale0,
		Tensor &weight1,
		Tensor &zp1,
		Tensor &scale1,
		Tensor &weight2,
		Tensor &zp2,
		Tensor &scale2,
		int64_t group_size,
		int64_t weight_bits,
		Tensor &output)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(weight0);
		CHECK_TENSOR_IN_DEVICE(zp0);
		CHECK_TENSOR_IN_DEVICE(scale0);
		CHECK_TENSOR_IN_DEVICE(weight1);
		CHECK_TENSOR_IN_DEVICE(zp1);
		CHECK_TENSOR_IN_DEVICE(scale1);
		CHECK_TENSOR_IN_DEVICE(weight2);
		CHECK_TENSOR_IN_DEVICE(zp2);
		CHECK_TENSOR_IN_DEVICE(scale2);
		CHECK_TENSOR_IN_DEVICE(output);

		TIMING_START;
#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLLamaA16MlpAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, weight0),
			tpu::TPUGenerateTpudnnTensor( stream, zp0),
			tpu::TPUGenerateTpudnnTensor( stream, scale0),
			tpu::TPUGenerateTpudnnTensor( stream, weight1),
			tpu::TPUGenerateTpudnnTensor( stream, zp1),
			tpu::TPUGenerateTpudnnTensor( stream, scale1),
			tpu::TPUGenerateTpudnnTensor( stream, weight2),
			tpu::TPUGenerateTpudnnTensor( stream, zp2),
			tpu::TPUGenerateTpudnnTensor( stream, scale2),
			group_size,
			weight_bits,
			tpu::TPUGenerateTpudnnTensor( stream,output));
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END(tpu::LLAMA_A16_MLP_FORWARD);
		return output;
	}

}
