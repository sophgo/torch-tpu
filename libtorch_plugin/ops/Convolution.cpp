#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

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

Tensor conv2d_tpu ( const Tensor & input_,
                    const Tensor & weight,
                    const c10::optional<Tensor> & bias_opt,
                    IntArrayRef stride,
                    IntArrayRef padding,
                    IntArrayRef dilation,
                    int64_t groups )
{
  CHECK_TENSOR_IN_DEVICE ( input_ );
  CHECK_TENSOR_IN_DEVICE ( weight );
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
  Tensor input;
  bool is_batched;
  std::tie ( input, is_batched ) =
  batchify ( input_, /*num_spatial_dims=*/ 2, "conv2d" );
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
    auto XDesc = tpu::TPUGenerateTensorDesc ( input_ );
    auto ConvOutputShape = at::native::conv_output_size (
                           input_.sizes(),
                           weight.sizes(),
                           padding,
                           stride,
                           dilation );
    Tensor Output = torch::empty ( ConvOutputShape,
                                   tpu::TPUGetCurrentDevice() );
    auto YDesc = tpu::TPUGenerateTensorDesc ( Output );
    FilterDescriptor_t WDesc =
    {
      .oc = ( int ) weight.size ( 0 ),
      .ic = ( int ) weight.size ( 1 ),
      .kh = ( int ) weight.size ( 2 ),
      .kw = ( int ) weight.size ( 3 ),
      .dtype = ( sg_data_type_t ) tpu::TPUConvertDType ( weight.dtype() )
    };
#if 0
    Status = sgdnn_conv_forward_cudnn ( Handle,
                                        &Alpha,
                                        XDesc,
                                        ADDR_IN_DEVICE ( input_ ),
                                        const FilterDescriptor_t        wDesc,
                                        ADDR_IN_DEVICE ( weight ),
                                        const ConvolutionDescriptor_t   convDesc,
                                        &Beta,
                                        YDesc,
                                        ADDR_IN_DEVICE ( Output ) );
#endif
    LOG ( FATAL ) << "TPU convolution is implemented";
  }
  return is_batched ? output : output.squeeze ( 0 );
}

TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "conv2d", conv2d_tpu );
}
TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
{
  m.impl ( "conv2d", conv2d_tpu );
}
} // namespace at
