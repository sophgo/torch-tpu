#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & binary_Tensor_tpu ( const Tensor          & input1,
                             const Tensor          & input2,
                             const Scalar          & alpha,
                             Tensor                & out,
                             const EltwiseOpMode_t & op )
{
  if ( input1.device().type() == DeviceType::PrivateUse1 &&
       input2.device().type() == DeviceType::PrivateUse1 )
  {
    CHECK_TENSOR_IN_DEVICE ( input1 );
    CHECK_TENSOR_IN_DEVICE ( input2 );
    CHECK_TENSOR_IN_DEVICE ( out );
    auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
    auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
    auto output_desc = tpu::TPUGenerateTensorDesc ( out );
    auto handle = tpu::TPUGetDeviceHandle();
    float alpha1 = 1.f;
    float alpha2 = alpha.toDouble();
    float beta = 0.f;
    bm_status_t status = BM_SUCCESS;
    status = sgdnn_eltwise_forward_cudnn ( handle,
                                           &alpha1,
                                           input1_desc,
                                           ADDR_IN_DEVICE ( input1 ),
                                           &alpha2,
                                           input2_desc,
                                           ADDR_IN_DEVICE ( input2 ),
                                           &beta,
                                           output_desc,
                                           ADDR_IN_DEVICE ( out ),
                                           op );
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
                          Tensor       & out )
{
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto input1_cpu = input1.to ( torch::Device ( "cpu" ) );
  auto input2_cpu = input2.to ( torch::Device ( "cpu" ) );
  auto output_exp = torch::add ( input1_cpu, input2_cpu, alpha );
#endif
  EltwiseOpMode_t op = OP_ELTWISE_COEF_ADD;
  auto & output = binary_Tensor_tpu ( input1, input2, alpha, out, op );
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto output_got = output.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( output_got, output_exp );
#endif
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "add.out", add_Tensor_tpu );
}
} // namespace at
