#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor llama_attention(
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &Kcache,
		Tensor &Vcache,
		Tensor &cos,
		Tensor &sin,
		const c10::optional<Tensor> &mask,
		Tensor &Y,
		int64_t embeddings,
		int64_t attention_mode,
		double C)
	{
		CHECK_TENSOR_IN_DEVICE(Q);
		CHECK_TENSOR_IN_DEVICE(K);
		CHECK_TENSOR_IN_DEVICE(V);
		CHECK_TENSOR_IN_DEVICE(Kcache);
		CHECK_TENSOR_IN_DEVICE(Vcache);
		CHECK_TENSOR_IN_DEVICE(cos);
		CHECK_TENSOR_IN_DEVICE(sin);
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());
		CHECK_TENSOR_IN_DEVICE(Y);

		TIMING_START;
		#if defined BACKEND_SG2260
		tpuRtStatus_t status = sgdnnLlamaAttention(
			tpu::TPUGetDeviceResource(),
			tpu::TPUGenerateSgdnnTensor(Q),
			tpu::TPUGenerateSgdnnTensor(K),
			tpu::TPUGenerateSgdnnTensor(V),
			tpu::TPUGenerateSgdnnTensor(Kcache),
			tpu::TPUGenerateSgdnnTensor(Vcache),
			tpu::TPUGenerateSgdnnTensor(cos),
			tpu::TPUGenerateSgdnnTensor(sin),
			mask.has_value() ? tpu::TPUGenerateSgdnnTensor(mask.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(Y),
			embeddings,
			attention_mode,
			C);
		TORCH_CHECK(status == tpuRtSuccess);
		#elif defined BACKEND_1684X
		TORCH_CHECK(false);
		#endif
		TIMING_END(tpu::LLAMA_ATTENTION);
		return Y;
	}

}
