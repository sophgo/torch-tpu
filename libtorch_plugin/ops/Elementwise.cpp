#include <torch/library.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & add_Tensor_tpu ( const Tensor & input1,
                          const Tensor & input2,
                          const Scalar & alpha,
                          Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( input1 );
  CHECK_TENSOR_IN_DEVICE ( input2 );
  if ( input1.dtype() == caffe2::TypeMeta::Make<float>() &&
       input2.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    //LOG ( FATAL ) << "Not implemented";
  }
  else
  {
    LOG ( FATAL ) << "Unsupported data type";
  }
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "add.out", add_Tensor_tpu );
}
}
