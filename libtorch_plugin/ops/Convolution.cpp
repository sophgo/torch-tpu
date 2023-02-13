#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_LIBTORCH_OP_COMPARE TRUE

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
  TORCH_CHECK ( at::isComplexType ( input_.scalar_type() ) == false,
                "Complex convolution is unsupported by TPU" );
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
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto weight_cpu = weight.to ( torch::Device ( "cpu" ) );
  Tensor bias_cpu;
  if ( bias.defined() )
  {
    bias_cpu = bias.to ( torch::Device ( "cpu" ) );
  }
  auto output_exp = torch::convolution (
                    input_cpu,
                    weight_cpu,
                    c10::optional<Tensor> ( bias_cpu ),
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups );
#endif
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
    output = torch::empty ( output_shape, tpu::TPUGetCurrentDevice() );
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
  }
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto output_got = output.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing convolution:"
            << " input shape = " << input.sizes()
            << " input dtype = " << input.dtype()
            << " weight shape = " << weight.sizes()
            << " weight dtype = " << weight.dtype()
            << " bias shape = " << bias.sizes()
            << " bias dtype = " << bias.dtype()
            << " output shape = " << output.sizes()
            << " output dtype = " << output.dtype()
            << " stride = " << stride
            << " padding = " << padding
            << " dilation = " << dilation
            << " groups = " << groups
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
#endif
  return is_batched ? output : output.squeeze ( 0 );
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
  TORCH_CHECK ( at::isComplexType ( input.scalar_type() ) == false,
                "Complex convolution backward is unsupported by TPU" );
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( weight );
  auto num_spatial_dims = weight.dim() - 2;
  Tensor grad_input, grad_weight, grad_bias;
  if ( output_mask[0] == true )
  {
    grad_input = empty ( input.sizes(), tpu::TPUGetCurrentDevice() );
  }
  if ( output_mask[1] == true )
  {
    grad_weight = empty ( weight.sizes(), tpu::TPUGetCurrentDevice() );
  }
  if ( output_mask[2] == true )
  {
    grad_bias = empty ( { weight.size ( 0 ) }, tpu::TPUGetCurrentDevice() );
  }
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto weight_cpu = weight.to ( torch::Device ( "cpu" ) );
  auto grad_output_cpu = grad_output.to ( torch::Device ( "cpu" ) );
  at::OptionalIntArrayRef bias_sizes_opt ( { weight.size ( 0 ) } );
  auto outputs_exp = torch::convolution_backward (
                     grad_output_cpu,
                     input_cpu,
                     weight_cpu,
                     bias_sizes_opt,
                     stride,
                     padding,
                     dilation,
                     transposed,
                     output_padding,
                     groups,
                     output_mask );
#endif
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
      .computeType = ( sg_data_type_t ) tpu::TPUConvertDType ( input.dtype() )
    };
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
  }
#ifdef TPU_LIBTORCH_OP_COMPARE
  std::cout << "Comparing convolution backward:"
            << " grad_output shape = " << grad_output.sizes()
            << " grad_output dtype = " << grad_output.dtype()
            << " input shape = " << input.sizes()
            << " input dtype = " << input.dtype()
            << " weight shape = " << weight.sizes()
            << " weight dtype = " << weight.dtype()
            << " grad_input shape = " << grad_input.sizes()
            << " grad_input dtype = " << grad_input.dtype()
            << " grad_weight shape = " << grad_weight.sizes()
            << " grad_weight dtype = " << grad_weight.dtype()
            << " grad_bias shape = " << grad_bias.sizes()
            << " grad_bias dtype = " << grad_bias.dtype()
            << " stride = " << stride
            << " padding = " << padding
            << " dilation = " << dilation
            << " groups = " << groups
            << std::endl;
  if ( output_mask[1] == true )
  {
    std::cout << "Compare grad_weight\n";
    auto grad_weight_got = grad_weight.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( grad_weight_got, std::get<1> ( outputs_exp ) );
  }
  if ( output_mask[2] == true )
  {
    std::cout << "Compare grad_bias\n";
    auto grad_bias_got = grad_bias.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( grad_bias_got, std::get<2> ( outputs_exp ) );
  }
  if ( output_mask[0] == true )
  {
    std::cout << "Compare grad_input\n";
    auto grad_input_got = grad_input.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( grad_input_got, std::get<0> ( outputs_exp ) );
  }
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "convolution_backward_overrideable",
           convolution_backward_overrideable_tpu );
}
} // namespace at
