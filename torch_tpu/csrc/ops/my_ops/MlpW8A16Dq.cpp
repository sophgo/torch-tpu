#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor mlp_w8a16_dq_forward(
		Tensor &input,
		Tensor &gate_weight,
		Tensor &up_weight,
		Tensor &down_weight,
        Tensor &gate_scale,
		Tensor &up_scale,
		Tensor &down_scale,
		Tensor &output,
		int64_t blocksize)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(gate_weight);
		CHECK_TENSOR_IN_DEVICE(up_weight);
		CHECK_TENSOR_IN_DEVICE(down_weight);
        CHECK_TENSOR_IN_DEVICE(gate_scale);
		CHECK_TENSOR_IN_DEVICE(up_scale);
		CHECK_TENSOR_IN_DEVICE(down_scale);
		CHECK_TENSOR_IN_DEVICE(output);

		TIMING_START;
#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnMlpW8A16DqAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, gate_weight),
			tpu::TPUGenerateTpudnnTensor( stream, up_weight),
			tpu::TPUGenerateTpudnnTensor( stream, down_weight),
            tpu::TPUGenerateTpudnnTensor( stream, gate_scale),
			tpu::TPUGenerateTpudnnTensor( stream, up_scale),
			tpu::TPUGenerateTpudnnTensor( stream, down_scale),
			tpu::TPUGenerateTpudnnTensor( stream, output),
            blocksize);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END(tpu::MLP_W8A16_DQ_FORWARD);
		return output;
	}

}
