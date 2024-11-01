#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor llava_attention(
	    Tensor &OUT, // (b, tokens, heads, heads_size)
	    Tensor &Q, // (tokens, heads, heads_size)
	    Tensor &K, // (tokens, heads, heads_size)
	    Tensor &V, // (tokens, heads, heads_size)
	    const c10::optional<Tensor> &mask,
	    double C)
	{
		CHECK_TENSOR_IN_DEVICE(OUT);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(Q);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(K);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(V);
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());

		TIMING_START;
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnMultiHeadAttentionAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor(stream, OUT),
			tpu::TPUGenerateTpudnnTensor(stream, Q),
			tpu::TPUGenerateTpudnnTensor(stream, K),
			tpu::TPUGenerateTpudnnTensor(stream, V),
			mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value()) : tpudnnUndefinedTensor(),
			tpudnnUndefinedTensor(),//Qbuffer
			tpudnnUndefinedTensor(),//Kbuffer
			tpudnnUndefinedTensor(),//Vbuffer
			C);
		TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS);
		TIMING_END(tpu::LLAVA_ATTENTION);
		return OUT;
	}
}
