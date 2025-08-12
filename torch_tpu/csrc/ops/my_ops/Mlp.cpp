#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor mlp_forward(
		Tensor &input,
		Tensor &w1,
		Tensor &w2,
		const c10::optional<Tensor> &b1,
		const c10::optional<Tensor> &b2,
		Tensor &out1,
		Tensor &p,
		Tensor &out2)
	{
		TIMING_START;
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w1);
		CHECK_TENSOR_IN_DEVICE(w2);
		if (b1.has_value())
			CHECK_TENSOR_IN_DEVICE(b1.value());
		if (b2.has_value())
			CHECK_TENSOR_IN_DEVICE(b2.value());
		CHECK_TENSOR_IN_DEVICE(out1);
		CHECK_TENSOR_IN_DEVICE(p);
		CHECK_TENSOR_IN_DEVICE(out2);
        // It seems that is tensors p and out1 are not used in the kernel
        // The function of llavaMlpAsync is the same as the API of sgdnnMlp
        auto handle = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnLLaVaMlpAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, input),
            tpu::TPUGenerateTpudnnTensor(handle, w1),
            tpu::TPUGenerateTpudnnTensor(handle, w2),
            b1.has_value() ? tpu::TPUGenerateTpudnnTensor(handle, b1.value()) : tpudnnUndefinedTensor(),
            b2.has_value()? tpu::TPUGenerateTpudnnTensor(handle, b2.value()) : tpudnnUndefinedTensor(),
            tpu::TPUGenerateTpudnnTensor(handle, out2));
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        TIMING_END;
		return out2;
	}
}
