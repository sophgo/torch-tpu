#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"


namespace at
{
    std::tuple<Tensor, Tensor, Tensor> ln_mm_forward(
        Tensor &input,
        Tensor &w,
		const c10::optional<Tensor> &b,
		Tensor &gamma,
		Tensor &beta,
        double eps,
        Tensor &mean,
        Tensor &rstd,
		Tensor &out
    )
    {
        CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(w);
		if (b.has_value())
			CHECK_TENSOR_IN_DEVICE(b.value());
		CHECK_TENSOR_IN_DEVICE(gamma);
		CHECK_TENSOR_IN_DEVICE(beta);
		CHECK_TENSOR_IN_DEVICE(mean);
		CHECK_TENSOR_IN_DEVICE(rstd);
		CHECK_TENSOR_IN_DEVICE(out);

		TIMING_START;
		#if defined BACKEND_SG2260
		tpuRtStatus_t status = sgdnnLnMm(
			c10_tpu::getCurrentTPUStream(),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(w),
			b.has_value() ? tpu::TPUGenerateSgdnnTensor(b.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(gamma),
			tpu::TPUGenerateSgdnnTensor(beta),
			eps,
			tpu::TPUGenerateSgdnnTensor(mean),
			tpu::TPUGenerateSgdnnTensor(rstd),
			tpu::TPUGenerateSgdnnTensor(out));
		TORCH_CHECK(status == tpuRtSuccess);
		#elif defined BACKEND_1684X
		TORCH_CHECK(false);
		#endif
		TIMING_END(tpu::LN_MM_FORWARD);
		return std::tuple<Tensor, Tensor, Tensor>(mean, rstd, out);
    }


	std::tuple<Tensor, Tensor, Tensor> ln_mm_backward(
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
		#if defined BACKEND_SG2260
		tpuRtStatus_t status = sgdnnLayernormBackward (
			c10_tpu::getCurrentTPUStream(),
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
		TORCH_CHECK(status == tpuRtSuccess);
		#elif defined BACKEND_1684X
		TORCH_CHECK(false);
		#endif
		TIMING_END(tpu::LN_MM_BACKWARD);
		return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_gamma, grad_beta);
    }
}