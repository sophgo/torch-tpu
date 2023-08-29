#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include <unistd.h>

#include "common/config.h"

namespace at
{

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_tpu (
const Tensor & input,
const c10::optional<Tensor> & weight_opt,
const c10::optional<Tensor> & bias_opt,
const c10::optional<Tensor> & running_mean_opt,
const c10::optional<Tensor> & running_var_opt,
bool training,
double momentum,
double eps )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "Batchnorm " << count << std::endl;
  ++count;
#endif
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = c10::value_or_else ( bias_opt, [] { return Tensor(); } );
  const Tensor & running_mean = c10::value_or_else ( running_mean_opt, [] { return Tensor(); } );
  const Tensor & running_var = c10::value_or_else ( running_var_opt, [] { return Tensor(); } );
  TORCH_CHECK ( training == true, "Batchnorm only supports training mode for now" );
  auto num_features = input.size ( 1 );
  CHECK_TENSOR_IN_DEVICE ( input );
  if ( weight.defined() )       { CHECK_TENSOR_IN_DEVICE ( weight ); }
  if ( bias.defined() )         { CHECK_TENSOR_IN_DEVICE ( bias ); }
  if ( running_mean.defined() ) { CHECK_TENSOR_IN_DEVICE ( running_mean ); }
  if ( running_var.defined() )  { CHECK_TENSOR_IN_DEVICE ( running_var ); }
#if 0
  auto running_mean_cpu = running_mean.defined() ? running_mean.cpu() : Tensor();
  auto running_var_cpu = running_var.defined() ? running_var.cpu() : Tensor();
  auto outputs_cpu = native_batch_norm (
                     TENSOR_TO_CPU ( input ),
                     c10::optional<Tensor> ( weight.defined() ? weight.cpu() : Tensor() ),
                     c10::optional<Tensor> ( bias.defined() ? bias.cpu() : Tensor() ),
                     c10::optional<Tensor> ( running_mean_cpu ),
                     c10::optional<Tensor> ( running_var_cpu ),
                     training,
                     momentum,
                     eps );
  if ( running_mean.defined() )
  {
    tpu::TPUCopyHostToDevice ( running_mean.data_ptr(), running_mean_cpu.contiguous().data_ptr(), running_mean.nbytes() );
  }
  if ( running_var.defined() )
  {
    tpu::TPUCopyHostToDevice ( running_var.data_ptr(), running_var_cpu.contiguous().data_ptr(), running_var.nbytes() );
  }
  return std::tuple<Tensor, Tensor, Tensor> (
         TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ),
         TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ),
         TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) );
#else
  auto output = torch::empty ( input.sizes(), input.options() );
  auto saved_mean = torch::empty ( { num_features }, input.options() );
  auto saved_invstd = torch::empty ( { num_features }, input.options() );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnBatchnorm2d (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( input ),
                       weight.defined() ? tpu::TPUGenerateSgdnnTensor ( weight ) : sgdnnUndefinedTensor(),
                       bias.defined() ? tpu::TPUGenerateSgdnnTensor ( bias ) : sgdnnUndefinedTensor(),
                       eps,
                       running_mean.defined() ? tpu::TPUGenerateSgdnnTensor ( running_mean ) : sgdnnUndefinedTensor(),
                       running_var.defined() ? tpu::TPUGenerateSgdnnTensor ( running_var ) : sgdnnUndefinedTensor(),
                       momentum,
                       tpu::TPUGenerateSgdnnTensor ( output ),
                       tpu::TPUGenerateSgdnnTensor ( saved_mean ),
                       tpu::TPUGenerateSgdnnTensor ( saved_invstd ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BATCHNORM, timer.ElapsedUS() );
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( output, saved_mean, saved_invstd );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "native_batch_norm", native_batch_norm_tpu );
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward_tpu (
const Tensor & grad_out,
const Tensor & input,
const c10::optional<Tensor> & weight_opt,
const c10::optional<Tensor> & running_mean_opt,
const c10::optional<Tensor> & running_var_opt,
const c10::optional<Tensor> & saved_mean_opt,
const c10::optional<Tensor> & saved_invstd_opt,
bool training,
double eps,
std::array<bool, 3> output_mask )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "Batchnorm Backward " << count << std::endl;
  ++count;
#endif
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & saved_mean = c10::value_or_else ( saved_mean_opt, [] { return Tensor(); } );
  const Tensor & saved_invstd = c10::value_or_else ( saved_invstd_opt, [] { return Tensor(); } );
  const Tensor & running_mean = c10::value_or_else ( running_mean_opt, [] { return Tensor(); } );
  const Tensor & running_var = c10::value_or_else ( running_var_opt, [] { return Tensor(); } );
  TORCH_CHECK ( training == true, "Batchnorm backward only supports training mode for now" );
  TORCH_CHECK ( weight.defined(), "weight must be defined" );
  TORCH_CHECK ( saved_mean.defined(), "saved_mean must be defined" );
  TORCH_CHECK ( saved_invstd.defined(), "saved_invstd must be defined" );
  auto num_features = input.size ( 1 );
  CHECK_TENSOR_IN_DEVICE ( grad_out );
  CHECK_TENSOR_IN_DEVICE ( input );
  if ( weight.defined() ) { CHECK_TENSOR_IN_DEVICE ( weight ); }
  if ( saved_mean.defined() ) { CHECK_TENSOR_IN_DEVICE ( saved_mean ); }
  if ( saved_invstd.defined() ) { CHECK_TENSOR_IN_DEVICE ( saved_invstd ); }
#if 0
  auto outputs_cpu = native_batch_norm_backward (
                     grad_out.cpu(),
                     input.cpu(),
                     c10::optional<Tensor> ( weight.defined() ? weight.cpu() : Tensor() ),
                     c10::optional<Tensor> ( running_mean.defined() ? running_mean.cpu() : Tensor() ),
                     c10::optional<Tensor> ( running_var.defined() ? running_var.cpu() : Tensor() ),
                     c10::optional<Tensor> ( saved_mean.defined() ? saved_mean.cpu() : Tensor() ),
                     c10::optional<Tensor> ( saved_invstd.defined() ? saved_invstd.cpu() : Tensor() ),
                     training,
                     eps,
                     output_mask );
  return std::tuple<Tensor, Tensor, Tensor> (
         output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ) : Tensor(),
         output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ) : Tensor(),
         output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) : Tensor() );
#else
  Tensor grad_input, grad_weight, grad_bias;
  if ( output_mask[0] == true )
  {
    grad_input = torch::empty ( input.sizes(), input.options() );
  }
  if ( output_mask[1] == true )
  {
    grad_weight = empty ( weight.sizes(), weight.options() );
  }
  if ( output_mask[2] == true )
  {
    // We assume that weight and bias have the same data type
    grad_bias = empty ( { weight.size ( 0 ) }, weight.options() );
  }
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnBatchnorm2dBackward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( grad_out ),
                       tpu::TPUGenerateSgdnnTensor ( input ),
                       weight.defined() ? tpu::TPUGenerateSgdnnTensor ( weight ) : sgdnnUndefinedTensor(),
                       saved_mean.defined() ? tpu::TPUGenerateSgdnnTensor ( saved_mean ) : sgdnnUndefinedTensor(),
                       saved_invstd.defined() ? tpu::TPUGenerateSgdnnTensor ( saved_invstd ) : sgdnnUndefinedTensor(),
                       output_mask[0] ? tpu::TPUGenerateSgdnnTensor ( grad_input ) : sgdnnUndefinedTensor(),
                       output_mask[1] ? tpu::TPUGenerateSgdnnTensor ( grad_weight ) : sgdnnUndefinedTensor(),
                       output_mask[2] ? tpu::TPUGenerateSgdnnTensor ( grad_bias ) : sgdnnUndefinedTensor() );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BATCHNORM_BACKWARD, timer.ElapsedUS() );
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "native_batch_norm_backward", native_batch_norm_backward_tpu );
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_tpu (
const Tensor &input,
IntArrayRef normalized_shape,
const c10::optional<at::Tensor> &weight_opt,
const c10::optional<at::Tensor> &bias_opt,
double eps )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "Layernorm " << count << std::endl;
  ++count;
#endif
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = c10::value_or_else ( bias_opt, [] { return Tensor(); } );
  auto input_ = input.contiguous();
  CHECK_TENSOR_IN_DEVICE ( input_ );
  if ( weight.defined() )       { CHECK_TENSOR_IN_DEVICE ( weight ); }
  if ( bias.defined() )         { CHECK_TENSOR_IN_DEVICE ( bias ); }
#if 0
  auto outputs_cpu = native_layer_norm (
                     TENSOR_TO_CPU ( input ),
                     c10::fromIntArrayRef ( normalized_shape )
                     c10::optional<Tensor> ( weight.defined() ? weight.cpu() : Tensor() ),
                     c10::optional<Tensor> ( bias.defined() ? bias.cpu() : Tensor() ),
                     eps );
  return std::tuple<Tensor, Tensor, Tensor> (
         TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ),
         TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ),
         TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) );
#else
  const auto input_shape = input_.sizes();
  const auto input_ndim = input_.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
  // input_shape = stat_shape + normalized_shape
  DimVector stat_shape;
  for ( const auto idx : c10::irange ( axis ) )
  {
    stat_shape.emplace_back ( input_shape[idx] );
  }
  for ( const auto idx C10_UNUSED : c10::irange ( axis, input_.dim() ) )
  {
    stat_shape.emplace_back ( 1 );
  }
  auto output = torch::empty ( input_shape, input_.options() );
  auto mean = torch::empty ( stat_shape, input_.options() );
  auto rstd = torch::empty ( stat_shape, input_.options() );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnLayernorm (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( input_ ),
                       weight.defined() ? tpu::TPUGenerateSgdnnTensor ( weight ) : sgdnnUndefinedTensor(),
                       bias.defined() ? tpu::TPUGenerateSgdnnTensor ( bias ) : sgdnnUndefinedTensor(),
                       axis,
                       eps,
                       tpu::TPUGenerateSgdnnTensor ( output ),
                       tpu::TPUGenerateSgdnnTensor ( mean ),
                       tpu::TPUGenerateSgdnnTensor ( rstd ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::LAYERNORM, timer.ElapsedUS() );
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( output, mean, rstd );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "native_layer_norm", native_layer_norm_tpu );
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_tpu (
const Tensor & grad_out,
const Tensor & input,
IntArrayRef normalized_shape,
const Tensor & mean,
const Tensor & rstd,
const c10::optional<Tensor> & weight_opt,
const c10::optional<Tensor> & bias_opt,
std::array<bool, 3> output_mask )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "Layernorm Backward " << count << std::endl;
  ++count;
#endif
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = c10::value_or_else ( bias_opt, [] { return Tensor(); } );
  CHECK_TENSOR_IN_DEVICE ( grad_out );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( mean );
  CHECK_TENSOR_IN_DEVICE ( rstd );
  if ( weight.defined() ) { CHECK_TENSOR_IN_DEVICE ( weight ); }
  if ( bias.defined() ) { CHECK_TENSOR_IN_DEVICE ( bias ); }
#if 0
  auto outputs_cpu = native_layer_norm_backward (
                     grad_out.cpu(),
                     input.cpu(),
                     normalized_shape,
                     mean.cpu(),
                     rstd.cpu(),
                     c10::optional<Tensor> ( weight.defined() ? weight.cpu() : Tensor() ),
                     c10::optional<Tensor> ( bias.defined() ? bias.cpu() : Tensor() ),
                     output_mask );
  return std::tuple<Tensor, Tensor, Tensor> (
         output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ) : Tensor(),
         output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ) : Tensor(),
         output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) : Tensor() );
#else
  Tensor grad_input, grad_weight, grad_bias;
  if ( output_mask[0] == true )
  {
    grad_input = torch::empty ( input.sizes(), input.options() );
  }
  if ( output_mask[1] == true )
  {
    grad_weight = empty ( weight.sizes(), weight.options() );
  }
  if ( output_mask[2] == true )
  {
    // We assume that weight and bias have the same data type
    grad_bias = empty ( { weight.size ( 0 ) }, weight.options() );
  }
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnLayernormBackward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( grad_out ),
                       tpu::TPUGenerateSgdnnTensor ( input ),
                       weight.defined() ? tpu::TPUGenerateSgdnnTensor ( weight ) : sgdnnUndefinedTensor(),
                       tpu::TPUGenerateSgdnnTensor ( mean ),
                       tpu::TPUGenerateSgdnnTensor ( rstd ),
                       axis,
                       output_mask[0] ? tpu::TPUGenerateSgdnnTensor ( grad_input ) : sgdnnUndefinedTensor(),
                       output_mask[1] ? tpu::TPUGenerateSgdnnTensor ( grad_weight ) : sgdnnUndefinedTensor(),
                       output_mask[2] ? tpu::TPUGenerateSgdnnTensor ( grad_bias ) : sgdnnUndefinedTensor() );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::LAYERNORM_BACKWARD, timer.ElapsedUS() );
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "native_layer_norm_backward", native_layer_norm_backward_tpu );
}
} // namespace at
