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
Tensor & pow_out_tpu( const Tensor & self, const Scalar & exponent, Tensor & out){
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 1
  LOG( WARNING ) << "pow_out_tpu use cpu impl";
  auto out_cpu = pow( self.cpu(), exponent );
  out = out_cpu.to(out.device());
#else
  // TODO
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "pow.Tensor_Scalar_out", pow_out_tpu );
}
} // namespace at
