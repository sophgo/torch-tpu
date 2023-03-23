#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & fill__Scalar_tpu ( Tensor & self, const Scalar & value )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  auto self_cpu = TENSOR_TO_CPU ( self );
  self_cpu.fill_ ( value );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
}
//TORCH_LIBRARY_IMPL ( aten, TPU, m )
//{
//  m.impl ( "fill_.Scalar", fill__Scalar_tpu );
//}

Tensor & zero__tpu ( Tensor & self )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  char * buffer = new char [self.nbytes()];
  memset ( buffer, 0x0, self.nbytes() );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), buffer, self.nbytes() );
  delete [] buffer;
  return self;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "zero_", zero__tpu );
}
} // namespace at
