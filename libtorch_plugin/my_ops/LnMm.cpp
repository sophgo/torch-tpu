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
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
		bm_status_t status = sgdnnLnMm(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(input),
			tpu::TPUGenerateSgdnnTensor(w),
			b.has_value() ? tpu::TPUGenerateSgdnnTensor(b.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(gamma),
			tpu::TPUGenerateSgdnnTensor(beta),
			eps,
			tpu::TPUGenerateSgdnnTensor(mean),
			tpu::TPUGenerateSgdnnTensor(rstd),
			tpu::TPUGenerateSgdnnTensor(out));
		TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::LN_MM_FORWARD, timer.ElapsedUS());
#endif

		return std::tuple<Tensor, Tensor, Tensor>(out, mean, rstd);
    }
}