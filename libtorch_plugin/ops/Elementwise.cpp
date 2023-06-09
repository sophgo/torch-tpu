#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & sqrt_out_tpu ( const Tensor & self, Tensor & out  )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  auto out_cpu = sqrt( self.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  return out;
}
// TORCH_LIBRARY_IMPL ( aten, TPU, m )
// {
//   m.impl ( "sqrt.out", sqrt_out_tpu );
// }
#if 0
Tensor & binary_Tensor_tpu ( const Tensor          & input1,
                             const Tensor          & input2,
                             const Scalar          & alpha,
                             Tensor                & out,
                             const EltwiseOpMode_t & op )
{
  if ( input1.device().type() == DeviceType::TPU && input2.device().type() == DeviceType::TPU )
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
    bm_status_t status = sgdnn_eltwise_forward_cudnn (
                         handle,
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
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else
  {
    LOG ( FATAL ) << "Inputs are all required in TPU device for now";
  }
  return out;
}
Tensor & add_out_tpu ( const Tensor & input1,
                       const Tensor & input2,
                       const Scalar & alpha,
                       Tensor       & out )
{
  static int count = 0;
  //std::cout << "Add " << count << std::endl;
#if 0
  auto out_cpu = torch::add ( input1.to ( torch::Device ( "cpu" ) ), input2.to ( torch::Device ( "cpu" ) ), alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  ++count;
  return out;
#else
  EltwiseOpMode_t op = OP_ELTWISE_COEF_ADD;
  auto & output = binary_Tensor_tpu ( input1, input2, alpha, out, op );
  ++count;
  return out;
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "add.out", add_out_tpu );
}

Tensor & sub_out_tpu ( const Tensor & input1,
                       const Tensor & input2,
                       const Scalar & alpha,
                       Tensor       & out )
{
  return add_out_tpu ( input1, input2, -alpha, out );
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "sub.out", sub_out_tpu );
}
#endif
} // namespace at
