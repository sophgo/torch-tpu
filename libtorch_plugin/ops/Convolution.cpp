#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

static std::tuple<Tensor, bool> batchify (
const Tensor & input,
const int64_t num_spatial_dims,
const std::string& func_name )
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
    LOG ( FATAL ) << "TPU Convolution2d is not implemented";
  }
  return is_batched ? output : output.squeeze ( 0 );
}

} // namespace at
