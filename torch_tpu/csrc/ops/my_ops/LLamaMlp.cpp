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
			tpu::TPUGetDeviceResource(),
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

	Tensor a16_llama_mlp_forward(
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
		tpuRtStatus_t status = sgdnnLLamaA16Mlp(
			tpu::TPUGetDeviceResource(),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(weight0),
			tpu::TPUGenerateSgdnnTensor(zp0),
			tpu::TPUGenerateSgdnnTensor(scale0),
			tpu::TPUGenerateSgdnnTensor(weight1),
			tpu::TPUGenerateSgdnnTensor(zp1),
			tpu::TPUGenerateSgdnnTensor(scale1),
			tpu::TPUGenerateSgdnnTensor(weight2),
			tpu::TPUGenerateSgdnnTensor(zp2),
			tpu::TPUGenerateSgdnnTensor(scale2),
			group_size,
			weight_bits,
			tpu::TPUGenerateSgdnnTensor(output));
		TORCH_CHECK(status == tpuRtSuccess);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END(tpu::LLAMA_A16_MLP_FORWARD);
		return output;
	}

}
