#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>

#include "TPUTorchUtils.h"
#include "TPUStorageImpl.h"
#include "common/config.h"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"
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

static bool check_output_shape_is_satified ( const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, IntArrayRef output_padding, int64_t groups )
{
  // check dtype if f32 return true
  if ( input.scalar_type() == at::kFloat )
  {
    return true;
  }
  auto num_spatial_dims = weight.dim() - 2;
  std::array<int64_t, 4> output_shape = {0, 0, 0, 0};
  output_shape[0] = input.size(0);
  output_shape[1] = weight.size(0);
  // 计算 output shape
  for (int64_t dim = 0; dim < num_spatial_dims; ++dim)
  {
      int64_t dim_size = (input.size(dim + 2) + 2 * padding[dim] - dilation[dim] * (weight.size(dim + 2) - 1) - 1) / stride[dim] + 1;
      output_shape[dim + 2] = dim_size;
  }
  // if output kh*kw  >= 4*16*1024 then return false
  // kh*kw need align to dtype size
  auto dtype_size = input.element_size();
  auto size = ((output_shape[0]-1)/64+1) * output_shape[2] * output_shape[3] * dtype_size;
  if ( size > 4 * 16 * 1024 )
  {
    return false;
  }
  return true;
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
  auto input_dtype = input_.scalar_type();
  auto output_cpu = torch::convolution ( input_.cpu().to( at::kFloat ),
                                         weight.cpu().to( at::kFloat ),
                                         c10::optional<Tensor> ( bias.defined() ? bias.cpu().to( at::kFloat ) : Tensor() ),
                                         stride,
                                         padding,
                                         dilation,
                                         transposed,
                                         output_padding,
                                         groups );
  output_cpu = output_cpu.to ( input_dtype );
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
    tpudnnConv2dParam_t conv_param =
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
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnConv2dAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, input),
      tpu::TPUGenerateTpudnnTensor(stream, weight),
      bias.defined() ? tpu::TPUGenerateTpudnnTensor (stream, bias ) : tpudnnUndefinedTensor(),
      conv_param,
      tpu::TPUGenerateTpudnnTensor(stream, output));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
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
  auto flag = check_output_shape_is_satified ( input, weight, stride, padding, dilation, output_padding, groups );
  if ( flag == false )
  {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto input_dtype = input.scalar_type();
    // change dtype into f32
    auto outputs_cpu = torch::convolution_backward (
                       grad_output.cpu().to ( at::kFloat ),
                       input.cpu().to ( at::kFloat ),
                       weight.cpu().to ( at::kFloat ),
                       at::OptionalIntArrayRef ( { weight.size ( 0 ) } ),
                       stride,
                       padding,
                       dilation,
                       transposed,
                       output_padding,
                       groups,
                       output_mask );
    TIMING_END ( tpu::CONVOLUTION_BACKWARD );
    Tensor grad_input, grad_weight, grad_bias;
    grad_input  = output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ).to ( input_dtype ) ) : Tensor();
    grad_weight = output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ).to ( input_dtype ) ) : Tensor();
    grad_bias   = output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ).to ( input_dtype ) ) : Tensor();
    SHOW_TENSOR_OP(grad_output_, input, weight, grad_input, grad_weight, grad_bias);
    return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias);
  }
#if 0
    CPU_IMPL_WARNING();
    TIMING_START;
    auto input_dtype = input.scalar_type();
    // change dtype into f32
    auto outputs_cpu = torch::convolution_backward (
                       grad_output.cpu().to ( at::kFloat ),
                       input.cpu().to ( at::kFloat ),
                       weight.cpu().to ( at::kFloat ),
                       at::OptionalIntArrayRef ( { weight.size ( 0 ) } ),
                       stride,
                       padding,
                       dilation,
                       transposed,
                       output_padding,
                       groups,
                       output_mask );
    TIMING_END ( tpu::CONVOLUTION_BACKWARD );
    Tensor grad_input, grad_weight, grad_bias;
    grad_input  = output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ).to ( input_dtype ) ) : Tensor();
    grad_weight = output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ).to ( input_dtype ) ) : Tensor();
    grad_bias   = output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ).to ( input_dtype ) ) : Tensor();
    SHOW_TENSOR_OP(grad_output_, input, weight, grad_input, grad_weight, grad_bias);
    return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias);
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
    if ( !at_tpu::StorageDescHelper::IsBaseFormatType(weight) )
      grad_weight = at_tpu::FormatCastPreparation::apply_tensor_with_format(weight, torch_tpu::TPU_DFORMAT_CONV_DW);
    else
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
    tpudnnConv2dParam_t conv_param =
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
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnConv2dBackwardAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, grad_output_),
      tpu::TPUGenerateTpudnnTensor(stream, input),
      tpu::TPUGenerateTpudnnTensor (stream, weight ),
      conv_param,
      output_mask[0] ? tpu::TPUGenerateTpudnnTensor (stream,  grad_input ) : tpudnnUndefinedTensor(),
      output_mask[1] ? tpu::TPUGenerateTpudnnTensor (stream,  grad_weight ) : tpudnnUndefinedTensor(),
      output_mask[2] ? tpu::TPUGenerateTpudnnTensor (stream,  grad_bias ) : tpudnnUndefinedTensor() );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
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
