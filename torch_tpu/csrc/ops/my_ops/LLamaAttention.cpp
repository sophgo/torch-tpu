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
	Tensor llama_attention(
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &Kcache,
		Tensor &Vcache,
		Tensor &weight1,
		Tensor &weight2,
		Tensor &weight3,
		Tensor &Y,
		double C)
	{
		CHECK_TENSOR_IN_DEVICE(Q);
		CHECK_TENSOR_IN_DEVICE(K);
		CHECK_TENSOR_IN_DEVICE(V);
		CHECK_TENSOR_IN_DEVICE(Kcache);
		CHECK_TENSOR_IN_DEVICE(Vcache);
		CHECK_TENSOR_IN_DEVICE(weight1);
		CHECK_TENSOR_IN_DEVICE(weight2);
		CHECK_TENSOR_IN_DEVICE(weight3);
		CHECK_TENSOR_IN_DEVICE(Y);
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
		bm_status_t status = sgdnnLlamaAttention(
			tpu::TPUGetDeviceHandle(),
			tpu::TPUGenerateSgdnnTensor(Q),
			tpu::TPUGenerateSgdnnTensor(K),
			tpu::TPUGenerateSgdnnTensor(V),
			tpu::TPUGenerateSgdnnTensor(Kcache),
			tpu::TPUGenerateSgdnnTensor(Vcache),
			tpu::TPUGenerateSgdnnTensor(weight1),
			tpu::TPUGenerateSgdnnTensor(weight2),
			tpu::TPUGenerateSgdnnTensor(weight3),
			tpu::TPUGenerateSgdnnTensor(Y),
			C);
		TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::LLAMA_ATTENTION, timer.ElapsedUS());
#endif
		return Y;
	}

}
