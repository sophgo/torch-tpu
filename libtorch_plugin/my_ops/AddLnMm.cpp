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
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
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
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::ADD_LN_MM_FORWARD, timer.ElapsedUS());
#endif

		return std::tuple<Tensor, Tensor, Tensor, Tensor>(out_add, mean, rstd, out);
    }
}