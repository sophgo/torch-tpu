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
Tensor & lerp_scalar_out_tpu(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
  CHECK_TENSOR_IN_DEVICE( out );
  CHECK_TENSOR_IN_DEVICE( self );
  CHECK_TENSOR_IN_DEVICE( end );
#if 1
    CPU_IMPL_WANING();
    auto out_cpu = lerp(self.cpu(), end.cpu(), weight);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.data_ptr(), out.nbytes());
#else
    TORCH_CHECK(false, "not impl lerp scalar kernel");
#endif
  SHOW_TENSOR_OP(self, end, out);
  return out;
}

Tensor & lerp_tensor_out_tpu(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
  CHECK_TENSOR_IN_DEVICE( out );
  CHECK_TENSOR_IN_DEVICE( self );
  CHECK_TENSOR_IN_DEVICE( end );
  CHECK_TENSOR_IN_DEVICE( weight );
#if 1
    CPU_IMPL_WANING();
    auto out_cpu = lerp(self.cpu(), end.cpu(), weight);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.data_ptr(), out.nbytes());
#else
    TORCH_CHECK(false, "not impl lerp tensor kernel");
#endif
  SHOW_TENSOR_OP(self, end, weight, out);
  return out;
}


TORCH_LIBRARY_IMPL (aten, TPU, m)
{
  m.impl ("lerp.Scalar_out", lerp_scalar_out_tpu);
  m.impl ("lerp.Tensor_out", lerp_tensor_out_tpu);
}


} // namespace at
