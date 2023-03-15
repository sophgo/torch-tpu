#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

Tensor & addcmul_out_tpu (
const Tensor & self,
const Tensor & tensor1,
const Tensor & tensor2,
const Scalar & value,
Tensor       & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( tensor1 );
  CHECK_TENSOR_IN_DEVICE ( tensor2 );
  CHECK_TENSOR_IN_DEVICE ( out );
  add_out ( out, self, tensor1 * tensor2, value );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "addcmul.out", addcmul_out_tpu );
}

} // namespace at
