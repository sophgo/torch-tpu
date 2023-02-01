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
  auto ADesc = tpu::TPUGenerateTensorDesc ( input1 );
  auto BDesc = tpu::TPUGenerateTensorDesc ( input2 );
  auto CDesc = tpu::TPUGenerateTensorDesc ( out );
  auto Handle = tpu::TPUGetDeviceHandle();
  float Alpha1 = 1.f;
  float Alpha2 = alpha.toDouble();
  float Beta = 0.f;
  OpTensorDescriptor_t Op = { .op_code = 1 };
  sgdnn_eltwise_forward ( Handle,
                          &Alpha1,
                          ADesc,
                          ADDR_IN_DEVICE ( input1 ),
                          &Alpha2,
                          BDesc,
                          ADDR_IN_DEVICE ( input2 ),
                          &Beta,
                          CDesc,
                          ADDR_IN_DEVICE ( out ),
                          Op );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "add.out", add_Tensor_tpu );
}
}
