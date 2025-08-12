#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor mm_w8a16_dq_forward(
		Tensor &input,
		Tensor &weight,
		Tensor &scale,
		Tensor &output,
		int64_t blocksize)
	{
		TIMING_START;
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(weight);
        CHECK_TENSOR_IN_DEVICE(scale);
		CHECK_TENSOR_IN_DEVICE(output);

#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnMmW8A16DqAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, weight),
            tpu::TPUGenerateTpudnnTensor( stream, scale),
			tpu::TPUGenerateTpudnnTensor( stream, output),
            blocksize);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END;
		return output;
	}

}
