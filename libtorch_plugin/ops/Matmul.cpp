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
  output = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "addmm.out", addmm_out_tpu );
}
} // namespace at
