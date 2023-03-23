#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
std::tuple<Tensor &, Tensor &> nll_loss_forward_output_tpu (
const Tensor                & input,
const Tensor                & target,
const c10::optional<Tensor> & weight_opt,
int64_t                       reduction,
int64_t                       ignore_index,
Tensor                      & output,
Tensor                      & total_weight )
{
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( target );
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( total_weight );
  if ( weight.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( weight );
  }
  auto input_cpu = TENSOR_TO_CPU ( input );
  auto target_cpu = TENSOR_TO_CPU ( target );
  Tensor weight_cpu;
  if ( weight.defined() )
  {
    weight_cpu = TENSOR_TO_CPU ( weight );
  }
  auto outputs_cpu = nll_loss_forward ( input_cpu, target_cpu, weight_cpu, reduction, ignore_index );
  output = TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) );
  total_weight = TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) );
  return std::tuple<Tensor &, Tensor &> ( output, total_weight );
}
//TORCH_LIBRARY_IMPL ( aten, TPU, m )
//{
//  m.impl ( "nll_loss_forward.output", nll_loss_forward_output_tpu );
//}

Tensor & nll_loss_backward_grad_input_tpu (
const Tensor                & grad_output,
const Tensor                & input,
const Tensor                & target,
const c10::optional<Tensor> & weight_opt,
int64_t                       reduction,
int64_t                       ignore_index,
const Tensor                & total_weight,
Tensor                      & grad_input )
{
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( target );
  CHECK_TENSOR_IN_DEVICE ( total_weight );
  CHECK_TENSOR_IN_DEVICE ( grad_input );
  if ( weight.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( weight );
  }
  auto grad_output_cpu = TENSOR_TO_CPU ( grad_output );
  auto input_cpu = TENSOR_TO_CPU ( input );
  auto target_cpu = TENSOR_TO_CPU ( target );
  auto total_weight_cpu = TENSOR_TO_CPU ( total_weight );
  Tensor weight_cpu;
  if ( weight.defined() )
  {
    weight_cpu = TENSOR_TO_CPU ( weight );
  }
  auto grad_input_cpu = nll_loss_backward ( grad_output_cpu, input_cpu, target_cpu, weight_cpu, reduction, ignore_index, total_weight_cpu );
  grad_input = TENSOR_TO_TPU ( grad_input_cpu );
  return grad_input;
}
//TORCH_LIBRARY_IMPL ( aten, TPU, m )
//{
//  m.impl ( "nll_loss_backward.grad_input", nll_loss_backward_grad_input_tpu );
//}
} // namespace at
