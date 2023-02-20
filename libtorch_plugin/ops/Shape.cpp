#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor _reshape_alias_tpu ( const Tensor & input,
                            IntArrayRef    sizes,
                            IntArrayRef    strides )
{
  CHECK_TENSOR_IN_DEVICE ( input );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto output_cpu = _reshape_alias ( input_cpu, sizes, strides );
  auto output = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_reshape_alias", _reshape_alias_tpu );
}

Tensor as_strided_tpu ( const Tensor         & input,
                        IntArrayRef            size,
                        IntArrayRef            stride,
                        c10::optional<int64_t> storage_offset )
{
  CHECK_TENSOR_IN_DEVICE ( input );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto output_cpu = as_strided ( input_cpu, size, stride, storage_offset ).contiguous();
  auto output = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "as_strided", as_strided_tpu );
}

Tensor view_tpu ( const Tensor & input, c10::IntArrayRef size )
{
  CHECK_TENSOR_IN_DEVICE ( input );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto output_cpu = input_cpu.view ( size );
  auto output = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "view", view_tpu );
}
} // namespace at
