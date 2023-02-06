#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define ERROR_CODE(Err) " ( TPU error code: " << Err << ")"

namespace at
{
Tensor & binary_Tensor_tpu ( const Tensor & input1,
                             const Tensor & input2,
                             const Scalar & alpha,
                             Tensor & out,
                             const OpTensorDescriptor_t & op )
{
  if ( input1.device().type() == DeviceType::PrivateUse1 &&
       input2.device().type() == DeviceType::PrivateUse1 )
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
    bm_status_t Status = BM_SUCCESS;
    Status = sgdnn_eltwise_forward ( Handle,
                                     &Alpha1,
                                     ADesc,
                                     ADDR_IN_DEVICE ( input1 ),
                                     &Alpha2,
                                     BDesc,
                                     ADDR_IN_DEVICE ( input2 ),
                                     &Beta,
                                     CDesc,
                                     ADDR_IN_DEVICE ( out ),
                                     op );
    if ( Status != BM_SUCCESS )
    {
      LOG ( FATAL ) << ERROR_CODE ( Status );
    }
  }
  else
  {
    LOG ( FATAL ) << "Inputs are all required in TPU device for now";
  }
  return out;
}
Tensor & add_Tensor_tpu ( const Tensor & input1,
                          const Tensor & input2,
                          const Scalar & alpha,
                          Tensor & out )
{
  OpTensorDescriptor_t Op = { .op_code = 1 };
  return binary_Tensor_tpu ( input1, input2, alpha, out, Op );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "add.out", add_Tensor_tpu );
}
}
