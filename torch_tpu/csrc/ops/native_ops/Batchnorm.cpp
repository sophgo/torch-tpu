#include <unistd.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{

static inline Tensor broadcast_like_channel(const Tensor& vecC, const Tensor& x) {
  // vecC shape [C] -> broadcast to [1, C, 1, 1, ...] for arithmetic with x
  std::vector<int64_t> shape(x.dim(), 1);
  shape[1] = vecC.size(0);
  return vecC.view(shape);
}

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
  TIMING_START;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = bias_opt.value_or(Tensor());
  const Tensor & running_mean = running_mean_opt.value_or(Tensor());
  const Tensor & running_var = running_var_opt.value_or(Tensor());
  auto num_features = input.size ( 1 );
  CHECK_TENSOR_IN_DEVICE ( input );
  if ( weight.defined() )       { CHECK_TENSOR_IN_DEVICE ( weight ); }
  if ( bias.defined() )         { CHECK_TENSOR_IN_DEVICE ( bias ); }
  if( !training ) {
    Tensor invstd = (running_var + eps).rsqrt();
    Tensor mean_bc = broadcast_like_channel(running_mean, input);
    Tensor invstd_bc = broadcast_like_channel(invstd, input);
    Tensor output = (input - mean_bc) * invstd_bc;

    if (weight.defined()) {
        output = output * broadcast_like_channel(weight, input);
    }
    if (bias.defined()) {
        output = output + broadcast_like_channel(bias, input);
    }
    TIMING_END;
    return std::tuple<Tensor, Tensor, Tensor> ( output, Tensor(), Tensor() );

    // auto dtype = input.scalar_type();
    // // implement inference mode
    // CPU_IMPL_WARNING("infer mode");
    // auto running_mean_cpu = running_mean.defined() ? running_mean.cpu().to(torch::kFloat) : Tensor();
    // auto running_var_cpu  = running_var.defined() ? running_var.cpu().to(torch::kFloat) : Tensor();
    // auto outputs_cpu = native_batch_norm (
    //                    input.cpu().to(torch::kFloat),
    //                    c10::optional<Tensor> ( weight.defined() ? weight.cpu().to(torch::kFloat) : Tensor() ),
    //                    c10::optional<Tensor> ( bias.defined() ? bias.cpu().to(torch::kFloat) : Tensor() ),
    //                    c10::optional<Tensor> ( running_mean_cpu ),
    //                    c10::optional<Tensor> ( running_var_cpu ),
    //                    training,
    //                    momentum,
    //                    eps );
    // TIMING_END;
    // return std::tuple<Tensor, Tensor, Tensor> (
    //         TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ).to(dtype) ),
    //         TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ).to(dtype) ),
    //         TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ).to(dtype) ) );
   }
  TORCH_CHECK ( training == true, "Batchnorm only supports training mode for now" );
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
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnBatchnorm2dAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, input),
      weight.defined() ? tpu::TPUGenerateTpudnnTensor(stream, weight) : tpudnnUndefinedTensor(),
      bias.defined() ? tpu::TPUGenerateTpudnnTensor(stream, bias) : tpudnnUndefinedTensor(),
      eps,
      running_mean.defined() ? tpu::TPUGenerateTpudnnTensor(stream, running_mean) : tpudnnUndefinedTensor(),
      running_var.defined() ? tpu::TPUGenerateTpudnnTensor(stream, running_var) : tpudnnUndefinedTensor(),
      momentum,
      tpu::TPUGenerateTpudnnTensor(stream, output),
      tpu::TPUGenerateTpudnnTensor(stream, saved_mean),
      tpu::TPUGenerateTpudnnTensor(stream, saved_invstd));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(input, weight, bias, running_mean, running_var, output, saved_mean, saved_invstd);
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
  TIMING_START;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & saved_mean = saved_mean_opt.value_or(Tensor());
  const Tensor & saved_invstd = saved_invstd_opt.value_or(Tensor());
  const Tensor & running_mean = running_mean_opt.value_or(Tensor());
  const Tensor & running_var = running_var_opt.value_or(Tensor());
  TORCH_CHECK ( training == true, "Batchnorm backward only supports training mode for now" );
  TORCH_CHECK ( weight.defined(), "weight must be defined" );
  TORCH_CHECK ( saved_mean.defined(), "saved_mean must be defined" );
  TORCH_CHECK ( saved_invstd.defined(), "saved_invstd must be defined" );
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
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnBatchnorm2dBackwardAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, grad_out),
      tpu::TPUGenerateTpudnnTensor(stream, input),
      weight.defined() ? tpu::TPUGenerateTpudnnTensor(stream, weight) : tpudnnUndefinedTensor(),
      saved_mean.defined() ? tpu::TPUGenerateTpudnnTensor(stream, saved_mean) : tpudnnUndefinedTensor(),
      saved_invstd.defined() ? tpu::TPUGenerateTpudnnTensor(stream, saved_invstd) : tpudnnUndefinedTensor(),
      output_mask[0] ? tpu::TPUGenerateTpudnnTensor(stream, grad_input) : tpudnnUndefinedTensor(),
      output_mask[1] ? tpu::TPUGenerateTpudnnTensor(stream, grad_weight) : tpudnnUndefinedTensor(),
      output_mask[2] ? tpu::TPUGenerateTpudnnTensor(stream, grad_bias) : tpudnnUndefinedTensor());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(grad_out, input, weight, saved_mean, saved_invstd, running_mean, running_var);
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
  TIMING_START;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = bias_opt.value_or(Tensor());

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
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnLayernormAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, input_),
      weight.defined() ? tpu::TPUGenerateTpudnnTensor(stream, weight) : tpudnnUndefinedTensor(),
      bias.defined() ? tpu::TPUGenerateTpudnnTensor(stream, bias) : tpudnnUndefinedTensor(),
      axis,
      eps,
      tpu::TPUGenerateTpudnnTensor(stream, output),
      tpu::TPUGenerateTpudnnTensor(stream, mean),
      tpu::TPUGenerateTpudnnTensor(stream, rstd));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(input, weight, bias);
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
  TIMING_START;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = bias_opt.value_or(Tensor());
  CHECK_TENSOR_IN_DEVICE ( grad_out );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( input ); //TODO: why not contiguous
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

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnLayernormBackwardAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, grad_out),
      tpu::TPUGenerateTpudnnTensor(stream, input.contiguous()),
      weight.defined() ? tpu::TPUGenerateTpudnnTensor(stream, weight) : tpudnnUndefinedTensor(),
      tpu::TPUGenerateTpudnnTensor(stream, mean),
      tpu::TPUGenerateTpudnnTensor(stream, rstd),
      axis,
      output_mask[0] ? tpu::TPUGenerateTpudnnTensor(stream, grad_input) : tpudnnUndefinedTensor(),
      output_mask[1] ? tpu::TPUGenerateTpudnnTensor(stream, grad_weight) : tpudnnUndefinedTensor(),
      output_mask[2] ? tpu::TPUGenerateTpudnnTensor(stream, grad_bias) : tpudnnUndefinedTensor(),
      output_mask[0] ? 1 : 0);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;

  SHOW_TENSOR_OP(grad_out, input, mean, rstd, grad_input, weight, bias);
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "native_layer_norm_backward", native_layer_norm_backward_tpu );
}
} // namespace at
