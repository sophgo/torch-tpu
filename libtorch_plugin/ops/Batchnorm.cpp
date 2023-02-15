#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

static inline void check_dims_match_num_input_features (
const char * arg_name,
SymInt       expected,
SymInt       actual )
{
  TORCH_CHECK ( actual == expected, arg_name, " should contain ", expected, " elements not ", actual );
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_tpu (
const Tensor                & input,
const c10::optional<Tensor> & weight_opt,
const c10::optional<Tensor> & bias_opt,
const c10::optional<Tensor> & running_mean_opt,
const c10::optional<Tensor> & running_var_opt,
bool                          training,
double                        momentum,
double                        eps )
{
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = c10::value_or_else ( bias_opt, [] { return Tensor(); } );
  const Tensor & running_mean = c10::value_or_else ( running_mean_opt, [] { return Tensor(); } );
  const Tensor & running_var = c10::value_or_else ( running_var_opt, [] { return Tensor(); } );
  TORCH_CHECK ( training == true, "Batchnorm only supports training mode for now" );
  TORCH_CHECK ( running_mean.defined(), "running_mean must be defined" );
  TORCH_CHECK ( running_var.defined(), "running_var must be defined" );
  auto num_features = input.sym_sizes() [1];
  CHECK_TENSOR_IN_DEVICE ( input );
  if ( weight.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( weight );
    check_dims_match_num_input_features ( "weight", num_features, weight.sym_numel() );
  }
  if ( bias.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( bias );
    check_dims_match_num_input_features ( "bias", num_features, bias.sym_numel() );
  }
  if ( running_mean.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( running_mean );
    check_dims_match_num_input_features ( "running_mean", num_features, running_mean.sym_numel() );
  }
  if ( running_var.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( running_var );
    check_dims_match_num_input_features ( "running_var", num_features, running_var.sym_numel() );
  }
  auto output_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( input.dtype() );
  auto output = torch::empty ( input.sizes(), output_options );
  auto save_mean = torch::empty_symint ( c10::SymIntArrayRef ( { num_features } ), output_options );
  auto save_invstd = torch::empty_symint ( c10::SymIntArrayRef ( { num_features } ), output_options );
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
  float alpha = 1.f;
  float beta = 0.f;
  auto input_desc = tpu::TPUGenerateTensorDesc ( input );
  auto output_desc = tpu::TPUGenerateTensorDesc ( output );
  status = sgdnn_batchnorm_forward_cudnn (
           handle,
           BatchNorm_Spatial,
           &alpha,
           &beta,
           );
  return std::tuple<Tensor, Tensor, Tensor> ( output, save_mean, save_invstd );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm", native_batch_norm_tpu );
}
} // namespace at
