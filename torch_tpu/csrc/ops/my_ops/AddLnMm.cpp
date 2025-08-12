#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "TPUStream.h"
#include "common/config.h"

namespace at
{
    std::tuple<Tensor, Tensor, Tensor, Tensor> add_ln_mm_forward(
        Tensor &input0,
        Tensor &input1,
        Tensor &w,
		const c10::optional<Tensor> &b,
		Tensor &gamma,
		Tensor &beta,
        double eps,
        Tensor &out_add,
        Tensor &mean,
        Tensor &rstd,
		Tensor &out
    )
    {
		TIMING_START;
        CHECK_TENSOR_IN_DEVICE(input0);
        CHECK_TENSOR_IN_DEVICE(input1);
		CHECK_TENSOR_IN_DEVICE(w);
		if (b.has_value())
			CHECK_TENSOR_IN_DEVICE(b.value());
		CHECK_TENSOR_IN_DEVICE(gamma);
		CHECK_TENSOR_IN_DEVICE(beta);
        CHECK_TENSOR_IN_DEVICE(out_add);
		CHECK_TENSOR_IN_DEVICE(mean);
		CHECK_TENSOR_IN_DEVICE(rstd);
		CHECK_TENSOR_IN_DEVICE(out);

        auto handle = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnFusedAddNormMMAsync(
			handle,
			tpu::TPUGenerateTpudnnTensor(handle, input0),
			tpu::TPUGenerateTpudnnTensor(handle, input1),
			tpu::TPUGenerateTpudnnTensor(handle, w),
			b.has_value()? tpu::TPUGenerateTpudnnTensor(handle, b.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor(handle, gamma),
			tpu::TPUGenerateTpudnnTensor(handle, beta),
			eps,
			tpu::TPUGenerateTpudnnTensor(handle, out_add),
			tpu::TPUGenerateTpudnnTensor(handle, mean),
			tpu::TPUGenerateTpudnnTensor(handle, rstd),
			tpu::TPUGenerateTpudnnTensor(handle, out));
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
		TIMING_END;
		return std::tuple<Tensor, Tensor, Tensor, Tensor>(out_add, mean, rstd, out);
    }

	std::tuple<Tensor, Tensor, Tensor> add_ln_mm_backward(
        Tensor &grad_out_ln,
        Tensor &input,
		Tensor &mean,
		Tensor &rstd,
		Tensor &gamma,
		Tensor &grad_input,
        Tensor &grad_gamma,
        Tensor &grad_beta
    )
    {
		TIMING_START;
        CHECK_TENSOR_IN_DEVICE(grad_out_ln);
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(mean);
		CHECK_TENSOR_IN_DEVICE(rstd);
		CHECK_TENSOR_IN_DEVICE(gamma);
		CHECK_TENSOR_IN_DEVICE(grad_input);
		CHECK_TENSOR_IN_DEVICE(grad_gamma);
		CHECK_TENSOR_IN_DEVICE(grad_beta);

        auto handle = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnLayernormBackwardAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, grad_out_ln),
            tpu::TPUGenerateTpudnnTensor(handle, input),
            tpu::TPUGenerateTpudnnTensor(handle, gamma),
            tpu::TPUGenerateTpudnnTensor(handle, mean),
            tpu::TPUGenerateTpudnnTensor(handle, rstd),
            -1,
            tpu::TPUGenerateTpudnnTensor(handle, grad_input),
            tpu::TPUGenerateTpudnnTensor(handle, grad_gamma),
            tpu::TPUGenerateTpudnnTensor(handle, grad_beta),
            true);
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        TIMING_END;
		return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_gamma, grad_beta);
    }
}