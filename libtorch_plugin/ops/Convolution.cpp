#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <TPUModule.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at
{

static std::tuple<Tensor, bool> batchify (
const Tensor      & input,
const int64_t       num_spatial_dims,
const std::string & func_name )
{
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = ( input.dim() == dim_count_batch );
  TORCH_CHECK ( input.dim() == dim_count_no_batch || is_batched,
                "Expected ", dim_count_no_batch, "D (unbatched) or ",
                dim_count_batch, "D (batched) input to ", func_name,
                ", but got input of size: ", input.sizes() );
  return std::make_tuple ( is_batched ? input : input.unsqueeze ( 0 ), is_batched );
}

Tensor convolution_tpu (
const Tensor                & input_,
const Tensor                & weight,
const c10::optional<Tensor> & bias_opt,
IntArrayRef                   stride,
IntArrayRef                   padding,
IntArrayRef                   dilation,
const bool                    transposed,
const IntArrayRef             output_padding,
int64_t                       groups )
{
  static int count = 0;
  //std::cout << "Convolution " << count << std::endl;
#if 0
  c10::MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor ( bias_opt );
  const Tensor & bias = *bias_maybe_owned;
  auto output_cpu = torch::convolution (
                    TENSOR_TO_CPU ( input_ ),
                    TENSOR_TO_CPU ( weight ),
                    c10::optional<Tensor> ( bias.defined() ? TENSOR_TO_CPU ( bias ) : Tensor() ),
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups );
  ++count;
  return TENSOR_TO_TPU ( output_cpu );
#else
  TORCH_CHECK ( at::isComplexType ( input_.scalar_type() ) == false, "Complex convolution is unsupported by TPU" );
  CHECK_TENSOR_IN_DEVICE ( input_ );
  CHECK_TENSOR_IN_DEVICE ( weight );
  c10::MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor ( bias_opt );
  const Tensor & bias = *bias_maybe_owned;
  TORCH_CHECK ( !bias.defined() || bias.dtype() == input_.dtype(),
                "Input type (", input_.dtype().name(), ") and bias type (",
                bias.dtype().name(), ") should be the same" );
  if ( bias.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( bias );
  }
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
    LOG ( FATAL ) << "TPU transposed convolution is not implemented";
  }
  else
  {
    if ( num_spatial_dims != 2 )
    {
      LOG ( FATAL ) << "TPU " << num_spatial_dims << "D convolution is not implemented";
    }
    auto handle = tpu::TPUGetDeviceHandle();
    bm_status_t status = BM_SUCCESS;
    float alpha = 1.f;
    float beta = 0.f;
    auto input_desc = tpu::TPUGenerateTensorDesc ( input );
    auto output_shape = at::native::conv_output_size (
                        input.sizes(),
                        weight.sizes(),
                        padding,
                        stride,
                        dilation );
    auto output_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( input.dtype() );
    output = torch::empty ( output_shape, output_options );
    auto output_desc = tpu::TPUGenerateTensorDesc ( output );
    FilterDescriptor_t weight_desc =
    {
      .oc = ( int ) weight.size ( 0 ),
      .ic = ( int ) weight.size ( 1 ),
      .kh = ( int ) weight.size ( 2 ),
      .kw = ( int ) weight.size ( 3 ),
      .dtype = ( sg_data_type_t ) tpu::TPUConvertDType ( weight.dtype() )
    };
    TensorDescriptor_t bias_desc;
    if ( bias.defined() )
    {
      bias_desc = tpu::TPUGenerateTensorDesc ( bias );
    }
    ConvolutionDescriptor_t conv_desc =
    {
      .pad_h = ( int ) padding[0],
      .pad_w = ( int ) padding[1],
      .stride_h = ( int ) stride[0],
      .stride_w = ( int ) stride[1],
      .dilation_h = ( int ) dilation[0],
      .dilation_w = ( int ) dilation[1],
      .groups = ( int ) groups,
      .computeType = ( sg_data_type_t ) tpu::TPUConvertDType ( input.dtype() )
    };
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    status = sgdnn_conv_forward_cudnn (
             handle,
             &alpha,
             input_desc,
             ADDR_IN_DEVICE ( input ),
             weight_desc,
             ADDR_IN_DEVICE ( weight ),
             bias_desc,
             bias.defined() ? ADDR_IN_DEVICE ( bias ) : nullptr,
             conv_desc,
             &beta,
             output_desc,
             ADDR_IN_DEVICE ( output ) );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CONVOLUTION, timer.ElapsedUS() );
#endif
  }
  ++count;
  return is_batched ? output : output.squeeze ( 0 );
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "convolution_overrideable", convolution_tpu );
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_tpu (
const Tensor      & grad_output,
const Tensor      & input,
const Tensor      & weight,
IntArrayRef         stride,
IntArrayRef         padding,
IntArrayRef         dilation,
bool                transposed,
IntArrayRef         output_padding,
int64_t             groups,
std::array<bool, 3> output_mask )
{
  static int count = 0;
  //std::cout << "Convolution Backward " << count << std::endl;
#if 0
  auto outputs_cpu =  torch::convolution_backward (
                      TENSOR_TO_CPU ( grad_output ),
                      TENSOR_TO_CPU ( input ),
                      TENSOR_TO_CPU ( weight ),
                      at::OptionalIntArrayRef ( { weight.size ( 0 ) } ),
                      stride,
                      padding,
                      dilation,
                      transposed,
                      output_padding,
                      groups,
                      output_mask );
  ++count;
  return std::tuple<Tensor, Tensor, Tensor> (
         output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ) : Tensor(),
         output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ) : Tensor(),
         output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) : Tensor() );
#else
  TORCH_CHECK ( at::isComplexType ( input.scalar_type() ) == false,
                "Complex convolution backward is unsupported by TPU" );
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  TORCH_CHECK ( grad_output.is_contiguous() );
  CHECK_TENSOR_IN_DEVICE ( input );
  TORCH_CHECK ( input.is_contiguous() );
  CHECK_TENSOR_IN_DEVICE ( weight );
  TORCH_CHECK ( weight.is_contiguous() );
  auto num_spatial_dims = weight.dim() - 2;
  Tensor grad_input, grad_weight, grad_bias;
  if ( output_mask[0] == true )
  {
    auto grad_input_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( input.dtype() );
    grad_input = empty ( input.sizes(), grad_input_options );
  }
  if ( output_mask[1] == true )
  {
    auto grad_weight_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( weight.dtype() );
    grad_weight = empty ( weight.sizes(), grad_weight_options );
  }
  if ( output_mask[2] == true )
  {
    // We assume that weight and bias have the same data type
    auto grad_bias_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( weight.dtype() );
    grad_bias = empty ( { weight.size ( 0 ) }, grad_bias_options );
  }
  if ( transposed == true )
  {
    LOG ( FATAL ) << "TPU transposed convolution backward is not implemented";
  }
  else
  {
    if ( num_spatial_dims != 2 )
    {
      LOG ( FATAL ) << "TPU " << num_spatial_dims << "D transposed convolution backward is not implemented";
    }
    auto handle = tpu::TPUGetDeviceHandle();
    bm_status_t status = BM_SUCCESS;
    float alpha = 1.f;
    float beta = 0.f;
    auto input_desc = tpu::TPUGenerateTensorDesc ( input );
    auto grad_output_desc = tpu::TPUGenerateTensorDesc ( grad_output );
    FilterDescriptor_t weight_desc =
    {
      .oc = ( int ) weight.size ( 0 ),
      .ic = ( int ) weight.size ( 1 ),
      .kh = ( int ) weight.size ( 2 ),
      .kw = ( int ) weight.size ( 3 ),
      .dtype = ( sg_data_type_t ) tpu::TPUConvertDType ( weight.dtype() )
    };
    TensorDescriptor_t grad_bias_desc;
    if ( output_mask[2] == true )
    {
      grad_bias_desc = tpu::TPUGenerateTensorDesc ( grad_bias );
    }
    ConvolutionDescriptor_t conv_desc =
    {
      .pad_h = ( int ) padding[0],
      .pad_w = ( int ) padding[1],
      .stride_h = ( int ) stride[0],
      .stride_w = ( int ) stride[1],
      .dilation_h = ( int ) dilation[0],
      .dilation_w = ( int ) dilation[1],
      .groups = ( int ) groups,
    };
    auto accuracy = tpu::GetConvolutionBackwardAccuracy();
    if ( accuracy == tpu::CONVOLUTION_BACKWARD_ACCURACY_FP32 )
    {
      conv_desc.computeType = SG_DTYPE_FP32;
    }
    else if ( accuracy == tpu::CONVOLUTION_BACKWARD_ACCURACY_FP16 )
    {
      conv_desc.computeType = SG_DTYPE_FP16;
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported convolution backward accuracy" );
    }
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    status = sgdnn_conv_backward_cudnn (
             handle,
             &alpha,
             &beta,
             input_desc,
             ADDR_IN_DEVICE ( input ),
             output_mask[0] ? ADDR_IN_DEVICE ( grad_input ) : nullptr,
             weight_desc,
             ADDR_IN_DEVICE ( weight ),
             output_mask[1] ? ADDR_IN_DEVICE ( grad_weight ) : nullptr,
             grad_bias_desc,
             output_mask[2] ? ADDR_IN_DEVICE ( grad_bias ) : nullptr,
             grad_output_desc,
             ADDR_IN_DEVICE ( grad_output ),
             conv_desc,
             output_mask[0],
             output_mask[1],
             output_mask[2] );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CONVOLUTION_BACKWARD, timer.ElapsedUS() );
#endif
  }
  ++count;
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "convolution_backward_overrideable", convolution_backward_overrideable_tpu );
}
} // namespace at
