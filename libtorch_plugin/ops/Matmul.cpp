#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & addmm_out_tpu ( const Tensor & input,
                         const Tensor & mat1,
                         const Tensor & mat2,
                         const Scalar & beta,
                         const Scalar & alpha,
                         Tensor       & output )
{
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( mat1 );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto mat1_cpu = mat1.to ( torch::Device ( "cpu" ) );
  auto mat2_cpu = mat2.to ( torch::Device ( "cpu" ) );
  auto output_cpu = addmm ( input_cpu, mat1_cpu, mat2_cpu, beta, alpha );
  tpu::TPUCopyHostToDevice ( output.data_ptr(), output_cpu.contiguous().data_ptr(), output.nbytes() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "addmm.out", addmm_out_tpu );
}

Tensor & mm_out_tpu ( const Tensor & mat1,
                      const Tensor & mat2,
                      Tensor       & output )
{
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( mat1 );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  auto mat1_cpu = mat1.to ( torch::Device ( "cpu" ) );
  auto mat2_cpu = mat2.to ( torch::Device ( "cpu" ) );
  auto output_cpu = mm ( mat1_cpu, mat2_cpu );
  tpu::TPUCopyHostToDevice ( output.data_ptr(), output_cpu.contiguous().data_ptr(), output.nbytes() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "mm.out", mm_out_tpu );
}

Tensor & bmm_out_tpu ( const Tensor & mat1,
                       const Tensor & mat2,
                       Tensor       & output )
{
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( mat1 );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  auto mat1_cpu = mat1.to ( torch::Device ( "cpu" ) );
  auto mat2_cpu = mat2.to ( torch::Device ( "cpu" ) );
  auto output_cpu = bmm ( mat1_cpu, mat2_cpu );
  tpu::TPUCopyHostToDevice ( output.data_ptr(), output_cpu.contiguous().data_ptr(), output.nbytes() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "bmm.out", bmm_out_tpu );
}

} // namespace at
