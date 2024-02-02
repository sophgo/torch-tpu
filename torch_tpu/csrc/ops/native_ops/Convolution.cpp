#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at
{

static std::tuple<Tensor, bool> batchify ( const Tensor & input, const int64_t num_spatial_dims, const std::string & func_name )
{
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = ( input.dim() == dim_count_batch );
  TORCH_CHECK ( input.dim() == dim_count_no_batch || is_batched, "Expected ", dim_count_no_batch, "D (unbatched) or ", dim_count_batch, "D (batched) input to ", func_name, ", but got input of size: ", input.sizes() );
  return std::make_tuple ( is_batched ? input : input.unsqueeze ( 0 ), is_batched );
}

Tensor convolution_tpu (
const Tensor & input_,
const Tensor & weight,
const c10::optional<Tensor> & bias_opt,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
const bool transposed,
const IntArrayRef output_padding,
int64_t groups )
{
  CHECK_TENSOR_IN_DEVICE ( input_ );
  CHECK_TENSOR_IN_DEVICE ( weight );
#if 0
  c10::MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor ( bias_opt );
  const Tensor & bias = *bias_maybe_owned;
  auto output_cpu = torch::convolution ( input_.cpu(),
                                         weight.cpu(),
                                         c10::optional<Tensor> ( bias.defined() ? bias.cpu() : Tensor() ),
                                         stride,
                                         padding,
                                         dilation,
                                         transposed,
                                         output_padding,
                                         groups );
  return TENSOR_TO_TPU ( output_cpu );
#else
  TORCH_CHECK ( at::isComplexType ( input_.scalar_type() ) == false, "Complex convolution is unsupported by TPU" );
  c10::MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor ( bias_opt );
  const Tensor & bias = *bias_maybe_owned;
  TORCH_CHECK ( !bias.defined() || bias.dtype() == input_.dtype(), "Input type (", input_.dtype().name(), ") and bias type (", bias.dtype().name(), ") should be the same" );
  if ( bias.defined() ) { CHECK_TENSOR_IN_DEVICE ( bias ); }
  auto num_spatial_dims = weight.dim() - 2;
  Tensor input;
  bool is_batched;
  auto func_name = "conv" + std::to_string ( num_spatial_dims ) + "d";
  if ( transposed == true )
  {
    func_name += "_transposed";
  }
  std::tie ( input, is_batched ) = batchify ( input_, num_spatial_dims, func_name );
  Tensor output;
  if ( transposed == true )
  {
    TORCH_CHECK ( false, "TPU transposed convolution is not implemented" );
  }
  else
  {
    TORCH_CHECK ( num_spatial_dims == 2, "TPU ", num_spatial_dims, "D convolution is not implemented" );
    auto output_shape = at::native::conv_output_size ( input.sizes(), weight.sizes(), padding, stride, dilation );
    output = torch::empty ( output_shape, input.options() );
    SgdnnConv2dParam_t conv_param =
    {
      .kernel_h = ( int ) weight.size ( 2 ),
      .kernel_w = ( int ) weight.size ( 3 ),
      .pad_h = ( int ) padding[0],
      .pad_w = ( int ) padding[1],
      .stride_h = ( int ) stride[0],
      .stride_w = ( int ) stride[1],
      .dilation_h = ( int ) dilation[0],
      .dilation_w = ( int ) dilation[1],
      .groups = ( int ) groups,
    };

    TIMING_START;
    #if defined BACKEND_1684X
    auto status = sgdnnConv2d (
                         tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor ( input ),
                         tpu::TPUGenerateSgdnnTensor ( weight ),
                         bias.defined() ? tpu::TPUGenerateSgdnnTensor ( bias ) : sgdnnUndefinedTensor(),
                         conv_param,
                         tpu::TPUGenerateSgdnnTensor ( output ) );
    TORCH_CHECK ( status == BM_SUCCESS );
    #elif defined BACKEND_SG2260
    auto status = sgdnnConv2d (
                         c10_tpu::getCurrentTPUStream(),
                         tpu::TPUGenerateSgdnnTensor ( input ),
                         tpu::TPUGenerateSgdnnTensor ( weight ),
                         bias.defined() ? tpu::TPUGenerateSgdnnTensor ( bias ) : sgdnnUndefinedTensor(),
                         conv_param,
                         tpu::TPUGenerateSgdnnTensor ( output ) );
    TORCH_CHECK ( status == tpuRtSuccess );
    #endif
    TIMING_END ( tpu::CONVOLUTION );
  }
  SHOW_TENSOR_OP(input_, weight, bias, output);
  return is_batched ? output : output.squeeze ( 0 );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "convolution_overrideable", convolution_tpu );
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_tpu (
const Tensor & grad_output,
const Tensor & input,
const Tensor & weight,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool transposed,
IntArrayRef output_padding,
int64_t groups,
std::array<bool, 3> output_mask )
{
  auto grad_output_ = grad_output.contiguous();
  CHECK_TENSOR_IN_DEVICE ( grad_output_ );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( weight );
#if 0
  auto outputs_cpu =  torch::convolution_backward (
                      grad_output.cpu(),
                      input.cpu(),
                      weight.cpu(),
                      at::OptionalIntArrayRef ( { weight.size ( 0 ) } ),
                      stride,
                      padding,
                      dilation,
                      transposed,
                      output_padding,
                      groups,
                      output_mask );
  return std::tuple<Tensor, Tensor, Tensor> (
         output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ) : Tensor(),
         output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ) : Tensor(),
         output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) : Tensor() );
#else
  TORCH_CHECK ( at::isComplexType ( input.scalar_type() ) == false,
                "Complex convolution backward is unsupported by TPU" );
  auto num_spatial_dims = weight.dim() - 2;
  Tensor grad_input, grad_weight, grad_bias;
  if ( output_mask[0] == true )
  {
    grad_input = empty ( input.sizes(), input.options() );
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
  if ( transposed == true )
  {
    TORCH_CHECK ( false, "TPU transposed convolution backward is not implemented" );
  }
  else
  {
    if ( num_spatial_dims != 2 )
    {
      TORCH_CHECK ( num_spatial_dims == 2, "TPU ", num_spatial_dims, "D transposed convolution backward is not implemented" );
    }
    SgdnnConv2dParam_t conv_param =
    {
      .kernel_h = ( int ) weight.size ( 2 ),
      .kernel_w = ( int ) weight.size ( 3 ),
      .pad_h = ( int ) padding[0],
      .pad_w = ( int ) padding[1],
      .stride_h = ( int ) stride[0],
      .stride_w = ( int ) stride[1],
      .dilation_h = ( int ) dilation[0],
      .dilation_w = ( int ) dilation[1],
      .groups = ( int ) groups,
    };
    TIMING_START;
    #if defined BACKEND_1684X
    auto status = sgdnnConv2dBackward (
                         tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor ( grad_output_ ),
                         tpu::TPUGenerateSgdnnTensor ( input ),
                         tpu::TPUGenerateSgdnnTensor ( weight ),
                         conv_param,
                         output_mask[0] ? tpu::TPUGenerateSgdnnTensor ( grad_input ) : sgdnnUndefinedTensor(),
                         output_mask[1] ? tpu::TPUGenerateSgdnnTensor ( grad_weight ) : sgdnnUndefinedTensor(),
                         output_mask[2] ? tpu::TPUGenerateSgdnnTensor ( grad_bias ) : sgdnnUndefinedTensor() );
    TORCH_CHECK ( status == BM_SUCCESS );
    #elif defined BACKEND_SG2260
    auto status = sgdnnConv2dBackward (
                         c10_tpu::getCurrentTPUStream(),
                         tpu::TPUGenerateSgdnnTensor ( grad_output_ ),
                         tpu::TPUGenerateSgdnnTensor ( input ),
                         tpu::TPUGenerateSgdnnTensor ( weight ),
                         conv_param,
                         output_mask[0] ? tpu::TPUGenerateSgdnnTensor ( grad_input ) : sgdnnUndefinedTensor(),
                         output_mask[1] ? tpu::TPUGenerateSgdnnTensor ( grad_weight ) : sgdnnUndefinedTensor(),
                         output_mask[2] ? tpu::TPUGenerateSgdnnTensor ( grad_bias ) : sgdnnUndefinedTensor() );
    TORCH_CHECK ( status == tpuRtSuccess );
    #endif
    TIMING_END ( tpu::CONVOLUTION_BACKWARD );
  }
  SHOW_TENSOR_OP(grad_output_, input, weight, grad_input, grad_weight, grad_bias);
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "convolution_backward_overrideable", convolution_backward_overrideable_tpu );
}
} // namespace at
