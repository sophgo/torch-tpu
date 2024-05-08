
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"
#include "ops.hpp"

namespace at {
Tensor &dummy(Tensor &in) {
  CHECK_TENSOR_IN_DEVICE(in);
  TIMING_START;
#ifdef BACKEND_SG2260
  SAFE_CALL(sgdnnDummy(tpu::TPUGetDeviceResource(), true));
#endif
  TIMING_END(tpu::DUMMY);
  return in;
}
Tensor &dummy_no_kernel_launch(Tensor &in) {
  CHECK_TENSOR_IN_DEVICE(in);
  TIMING_START;
#ifdef BACKEND_SG2260
  SAFE_CALL(sgdnnDummy_WO_KERNEL_LAUNCH(tpu::TPUGetDeviceResource(), true));
#endif
  TIMING_END(tpu::DUMMY);
  return in;
}
} // namespace at

namespace at
{
	TORCH_LIBRARY(my_ops, m)
	{
		m.def("dummy", dummy);
		m.def("dummy_no_kernel_launch", dummy_no_kernel_launch);
		m.def("mlp_forward", mlp_forward);
		m.def("mlp_backward", mlp_backward);
		m.def("llama_mlp_forward", llama_mlp_forward);
		m.def("a16_matmul_forward", a16_matmul_forward);
		m.def("a16_llama_mlp_forward", a16_llama_mlp_forward);
		m.def("rmsnorm_forward", rmsnorm_forward);
		m.def("llama_attention", llama_attention);
		m.def("attn_forward", attn_forward);
		m.def("attn_backward", attn_backward);
		m.def("ln_mm_forward", ln_mm_forward);
		m.def("ln_mm_backward", ln_mm_backward);
		m.def("add_ln_mm_forward", add_ln_mm_forward);
		m.def("add_ln_mm_backward", add_ln_mm_backward);
	}
}