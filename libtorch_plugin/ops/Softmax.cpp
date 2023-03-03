#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & _log_softmax_out_tpu ( const Tensor & input,
                                int64_t        dim,
                                bool           half_to_float,
                                Tensor       & output )
{
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( output );
  TORCH_CHECK ( half_to_float == false );
  auto input_cpu = TENSOR_TO_CPU ( input );
  auto output_cpu = log_softmax ( input_cpu, dim, c10::optional<ScalarType> ( input.scalar_type() ) );
  output = TENSOR_TO_TPU ( output_cpu );
  return output;
}
//TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
//{
//  m.impl ( "_log_softmax.out", _log_softmax_out_tpu );
//}

Tensor & _log_softmax_backward_data_out_tpu (
const Tensor & grad_output,
const Tensor & output,
int64_t        dim,
ScalarType     input_dtype,
Tensor       & out )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( out );
  auto grad_output_cpu = TENSOR_TO_CPU ( grad_output );
  auto output_cpu = TENSOR_TO_CPU ( output );
  auto out_cpu = _log_softmax_backward_data ( grad_output_cpu, output_cpu, dim, input_dtype );
  out = TENSOR_TO_TPU ( out_cpu );
  return out;
}
//TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
//{
//  m.impl ( "_log_softmax_backward_data.out", _log_softmax_backward_data_out_tpu );
//}
} // namespace at
