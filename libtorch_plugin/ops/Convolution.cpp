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
const Tensor & input,
const int64_t num_spatial_dims,
const std::string & func_name )
{
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = ( input.dim() == dim_count_batch );
  TORCH_CHECK ( input.dim() == dim_count_no_batch || is_batched,
                "Expected ",
                dim_count_no_batch,
                "D (unbatched) or ",
                dim_count_batch,
                "D (batched) input to ",
                func_name,
                ", but got input of size: ",
                input.sizes() );
  return std::make_tuple ( is_batched ? input : input.unsqueeze ( 0 ),
                           is_batched );
}

Tensor convolution_tpu ( const Tensor & input_,
                         const Tensor & weight,
                         const c10::optional<Tensor> & bias_opt,
                         IntArrayRef stride,
                         IntArrayRef padding,
                         IntArrayRef dilation,
                         const bool transposed,
                         const IntArrayRef output_padding,
                         int64_t groups )
{
  if ( transposed == true )
  {
    LOG ( FATAL ) << "TPU transposed convolution is not implemented";
  }
  else
  {
    CHECK_TENSOR_IN_DEVICE ( input_ );
    CHECK_TENSOR_IN_DEVICE ( weight );
    int convdims = weight.dim() - 2;
    if ( convdims != 2 )
    {
      LOG ( FATAL ) << convdims << "D convolution is unsupported by TPU";
    }
    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> bias_maybe_owned =
    at::borrow_from_optional_tensor ( bias_opt );
    const Tensor & bias = *bias_maybe_owned;
    TORCH_CHECK (
    !bias.defined() || bias.dtype() == input_.dtype(),
    "Input type (",
    input_.dtype().name(),
    ") and bias type (",
    bias.dtype().name(),
    ") should be the same" );
    if ( bias.defined() )
    {
      CHECK_TENSOR_IN_DEVICE ( bias );
    }
    Tensor input;
    bool is_batched;
    std::tie ( input, is_batched ) =
    batchify ( input_, /*num_spatial_dims=*/ 2, "conv2d" );
#ifdef TPU_LIBTORCH_OP_COMPARE
    auto InputCPU = input.to ( torch::Device ( "cpu" ) );
    auto WeightCPU = weight.to ( torch::Device ( "cpu" ) );
    at::Tensor BiasCPU;
    c10::optional<Tensor> BiasOpt;
    if ( bias.defined() )
    {
      BiasCPU = bias.to ( torch::Device ( "cpu" ) );
      BiasOpt = BiasCPU;
    }
    auto OutputExp = torch::conv2d ( InputCPU,
                                     WeightCPU,
                                     BiasOpt,
                                     stride,
                                     padding,
                                     dilation,
                                     groups );
#endif
    Tensor output;
    if ( at::isComplexType ( input_.scalar_type() ) )
    {
      LOG ( FATAL ) << "Complex convolution is unsupported";
    }
    else
    {
      auto Handle = tpu::TPUGetDeviceHandle();
      bm_status_t Status = BM_SUCCESS;
      float Alpha = 1.f;
      float Beta = 0.f;
      auto XDesc = tpu::TPUGenerateTensorDesc ( input );
      auto ConvOutputShape = at::native::conv_output_size (
                             input.sizes(),
                             weight.sizes(),
                             padding,
                             stride,
                             dilation );
      output = torch::empty ( ConvOutputShape,
                              tpu::TPUGetCurrentDevice() );
      auto YDesc = tpu::TPUGenerateTensorDesc ( output );
      FilterDescriptor_t WDesc =
      {
        .oc = ( int ) weight.size ( 0 ),
        .ic = ( int ) weight.size ( 1 ),
        .kh = ( int ) weight.size ( 2 ),
        .kw = ( int ) weight.size ( 3 ),
        .dtype = ( sg_data_type_t ) tpu::TPUConvertDType ( weight.dtype() )
      };
      TensorDescriptor_t BDesc;
      if ( bias.defined() )
      {
        tpu::TPUGenerateTensorDesc ( bias );
      }
      ConvolutionDescriptor_t ConvDesc =
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
      Status = sgdnn_conv_forward_cudnn (
               Handle,
               &Alpha,
               XDesc,
               ADDR_IN_DEVICE ( input ),
               WDesc,
               ADDR_IN_DEVICE ( weight ),
               BDesc,
               bias.defined() ? ADDR_IN_DEVICE ( bias ) : nullptr,
               ConvDesc,
               &Beta,
               YDesc,
               ADDR_IN_DEVICE ( output ) );
    }
#ifdef TPU_LIBTORCH_OP_COMPARE
    auto OutputGot = output.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( OutputGot, OutputExp );
#endif
    return is_batched ? output : output.squeeze ( 0 );
  }
}

TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
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
  LOG ( FATAL ) << "Convolution backward is unsupported by TPU";
}

TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "convolution_backward_overrideable",
           convolution_backward_overrideable_tpu );
}
} // namespace at
