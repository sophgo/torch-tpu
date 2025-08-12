#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor llava_mlp(
		Tensor &input,
		Tensor &w1,
		Tensor &w2,
		const c10::optional<Tensor> &b1,
		const c10::optional<Tensor> &b2,
		Tensor &out)
	{
		TIMING_START;
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w1);
		CHECK_TENSOR_IN_DEVICE(w2);
		if (b1.has_value())
			CHECK_TENSOR_IN_DEVICE(b1.value());
		if (b2.has_value())
			CHECK_TENSOR_IN_DEVICE(b2.value());
		CHECK_TENSOR_IN_DEVICE(out);
#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLLaVaMlpAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, w1),
			tpu::TPUGenerateTpudnnTensor( stream, w2),
			b1.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, (b1.value())) : tpudnnUndefinedTensor(),
			b2.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, (b2.value())) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor( stream, out));
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END;
		return out;
	}

}
