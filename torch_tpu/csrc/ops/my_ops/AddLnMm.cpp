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

		TIMING_START;
		bm_status_t status = sgdnnAddLnMm(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(input0),
            tpu::TPUGenerateSgdnnTensor(input1),
			tpu::TPUGenerateSgdnnTensor(w),
			b.has_value() ? tpu::TPUGenerateSgdnnTensor(b.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(gamma),
			tpu::TPUGenerateSgdnnTensor(beta),
			eps,
            tpu::TPUGenerateSgdnnTensor(out_add),
			tpu::TPUGenerateSgdnnTensor(mean),
			tpu::TPUGenerateSgdnnTensor(rstd),
			tpu::TPUGenerateSgdnnTensor(out));
		TORCH_CHECK(status == BM_SUCCESS);
		TIMING_END(tpu::ADD_LN_MM_FORWARD);
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
        CHECK_TENSOR_IN_DEVICE(grad_out_ln);
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(mean);
		CHECK_TENSOR_IN_DEVICE(rstd);
		CHECK_TENSOR_IN_DEVICE(gamma);
		CHECK_TENSOR_IN_DEVICE(grad_input);
		CHECK_TENSOR_IN_DEVICE(grad_gamma);
		CHECK_TENSOR_IN_DEVICE(grad_beta);

		TIMING_START;
		bm_status_t status = sgdnnLayernormBackward (
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor ( grad_out_ln ),
			tpu::TPUGenerateSgdnnTensor ( input ),
			tpu::TPUGenerateSgdnnTensor ( gamma ),
			tpu::TPUGenerateSgdnnTensor ( mean ),
			tpu::TPUGenerateSgdnnTensor ( rstd ),
			-1,
			tpu::TPUGenerateSgdnnTensor( grad_input ),
			tpu::TPUGenerateSgdnnTensor( grad_gamma ),
			tpu::TPUGenerateSgdnnTensor( grad_beta ),
			1);
		TORCH_CHECK(status == BM_SUCCESS);
		TIMING_END(tpu::ADD_LN_MM_BACKWARD);
		return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_gamma, grad_beta);
    }
}