#include <torch/library.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>

namespace at
{
Tensor & add_Tensor_tpu ( const Tensor & input1,
                          const Tensor & input2,
                          const Scalar & alpha,
                          Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( input1 );
  CHECK_TENSOR_IN_DEVICE ( input2 );
  LOG ( FATAL ) << "Not implemented";
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "add.out", add_Tensor_tpu );
}
}
