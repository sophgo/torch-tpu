#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

Tensor & addcmul_out_tpu ( const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( tensor1 );
  CHECK_TENSOR_IN_DEVICE ( tensor2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = addcmul ( self.cpu(), tensor1.cpu(), tensor2.cpu(), value );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  add_out ( out, self, tensor1 * tensor2, value );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "addcmul.out", addcmul_out_tpu );
}

} // namespace at
