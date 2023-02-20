#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & mean_out_tpu ( const Tensor              & input,
                        c10::OptionalArrayRef<long> dim_opt,
                        bool                        keepdim,
                        c10::optional<ScalarType>   dtype_opt,
                        Tensor                    & output )
{
  CHECK_TENSOR_IN_DEVICE ( input );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto output_cpu = mean ( input_cpu, dim_opt, keepdim, dtype_opt );
  output = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "mean.out", mean_out_tpu );
}
} // namespace at
