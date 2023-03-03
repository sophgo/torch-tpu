#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include <unistd.h>

#define TPU_OP_TIMING

namespace at
{

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
  static int count = 0;
  //std::cout << "Batchnorm " << count << std::endl;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = c10::value_or_else ( bias_opt, [] { return Tensor(); } );
  const Tensor & running_mean = c10::value_or_else ( running_mean_opt, [] { return Tensor(); } );
  const Tensor & running_var = c10::value_or_else ( running_var_opt, [] { return Tensor(); } );
  TORCH_CHECK ( training == true, "Batchnorm only supports training mode for now" );
  TORCH_CHECK ( weight.defined(), "weight must be defined" );
  TORCH_CHECK ( bias.defined(), "bias must be defined" );
  TORCH_CHECK ( running_mean.defined(), "running_mean must be defined" );
  TORCH_CHECK ( running_var.defined(), "running_var must be defined" );
  auto num_features = input.size ( 1 );
  CHECK_TENSOR_IN_DEVICE ( input );
  TORCH_CHECK ( input.is_contiguous() );
  if ( weight.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( weight );
  }
  if ( bias.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( bias );
  }
  if ( running_mean.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( running_mean );
  }
  if ( running_var.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( running_var );
  }
#if 0
  auto running_mean_cpu = running_mean.defined() ? TENSOR_TO_CPU ( running_mean ) : Tensor();
  auto running_var_cpu = running_var.defined() ? TENSOR_TO_CPU ( running_var ) : Tensor();
  auto outputs_cpu = native_batch_norm (
                     TENSOR_TO_CPU ( input ),
                     c10::optional<Tensor> ( weight.defined() ? TENSOR_TO_CPU ( weight ) : Tensor() ),
                     c10::optional<Tensor> ( bias.defined() ? TENSOR_TO_CPU ( bias ) : Tensor() ),
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
  ++count;
  return std::tuple<Tensor, Tensor, Tensor> (
         TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ),
         TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ),
         TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) );
#else
  auto output_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( input.dtype() );
  auto output = torch::empty ( input.sizes(), output_options );
  auto other_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( weight.dtype() );
  auto save_mean = torch::empty ( c10::IntArrayRef ( { num_features } ), other_options );
  auto save_invstd = torch::empty ( c10::IntArrayRef ( { num_features } ), other_options );
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
  float alpha = 1.f;
  float beta = 0.f;
  auto input_desc = tpu::TPUGenerateTensorDesc ( input );
  auto output_desc = tpu::TPUGenerateTensorDesc ( output );
  auto other_desc = tpu::TPUGenerateTensorDesc ( weight );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  status = sgdnn_batchnorm_forward_cudnn (
           handle,
           BatchNorm_Spatial,
           &alpha,
           &beta,
           input_desc,
           ADDR_IN_DEVICE ( input ),
           output_desc,
           ADDR_IN_DEVICE ( output ),
           other_desc,
           weight.defined() ? ADDR_IN_DEVICE ( weight ) : nullptr,
           bias.defined() ? ADDR_IN_DEVICE ( bias ) : nullptr,
           momentum,
           running_mean.defined() ? ADDR_IN_DEVICE ( running_mean ) : nullptr,
           running_var.defined() ? ADDR_IN_DEVICE ( running_var ) : nullptr,
           eps,
           ADDR_IN_DEVICE ( save_mean ),
           ADDR_IN_DEVICE ( save_invstd ) );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BATCHNORM, timer.ElapsedUS() );
#endif
  ++count;
  return std::tuple<Tensor, Tensor, Tensor> ( output, save_mean, save_invstd );
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm", native_batch_norm_tpu );
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward_tpu (
const Tensor                & grad_output,
const Tensor                & input,
const c10::optional<Tensor> & weight_opt,
const c10::optional<Tensor> & running_mean_opt,
const c10::optional<Tensor> & running_var_opt,
const c10::optional<Tensor> & save_mean_opt,
const c10::optional<Tensor> & save_invstd_opt,
bool                          training,
double                        eps,
std::array<bool, 3>           output_mask )
{
  static int count = 0;
  //std::cout << "Batchnorm Backward " << count << std::endl;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & save_mean = c10::value_or_else ( save_mean_opt, [] { return Tensor(); } );
  const Tensor & save_invstd = c10::value_or_else ( save_invstd_opt, [] { return Tensor(); } );
  const Tensor & running_mean = c10::value_or_else ( running_mean_opt, [] { return Tensor(); } );
  const Tensor & running_var = c10::value_or_else ( running_var_opt, [] { return Tensor(); } );
  TORCH_CHECK ( training == true, "Batchnorm backward only supports training mode for now" );
  TORCH_CHECK ( weight.defined(), "weight must be defined" );
  TORCH_CHECK ( save_mean.defined(), "save_mean must be defined" );
  TORCH_CHECK ( save_invstd.defined(), "save_invstd must be defined" );
  auto num_features = input.size ( 1 );
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( input );
  if ( weight.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( weight );
  }
  if ( save_mean.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( save_mean );
  }
  if ( save_invstd.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( save_invstd );
  }
#if 0
  auto outputs_cpu = native_batch_norm_backward (
                     TENSOR_TO_CPU ( grad_output ),
                     TENSOR_TO_CPU ( input ),
                     c10::optional<Tensor> ( weight.defined() ? TENSOR_TO_CPU ( weight ) : Tensor() ),
                     c10::optional<Tensor> ( running_mean.defined() ? TENSOR_TO_CPU ( running_mean ) : Tensor() ),
                     c10::optional<Tensor> ( running_var.defined() ? TENSOR_TO_CPU ( running_var ) : Tensor() ),
                     c10::optional<Tensor> ( save_mean.defined() ? TENSOR_TO_CPU ( save_mean ) : Tensor() ),
                     c10::optional<Tensor> ( save_invstd.defined() ? TENSOR_TO_CPU ( save_invstd ) : Tensor() ),
                     training,
                     eps,
                     output_mask );
  ++count;
  return std::tuple<Tensor, Tensor, Tensor> (
         output_mask[0] ? TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ) : Tensor(),
         output_mask[1] ? TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ) : Tensor(),
         output_mask[2] ? TENSOR_TO_TPU ( std::get<2> ( outputs_cpu ) ) : Tensor() );
#else
  Tensor grad_input, grad_weight, grad_bias;
  if ( output_mask[0] == true )
  {
    auto grad_input_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( input.dtype() );
    grad_input = torch::empty ( input.sizes(), grad_input_options );
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
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
  float alpha_data_diff = 1.f;
  float beta_data_diff = 0.f;
  float alpha_param_diff = 1.f;
  float beta_param_diff = 0.f;
  auto grad_output_desc = tpu::TPUGenerateTensorDesc ( grad_output );
  auto input_desc = tpu::TPUGenerateTensorDesc ( input );
  TensorDescriptor_t grad_input_desc, grad_weight_bias_desc;
  if ( output_mask[0] == true )
  {
    grad_input_desc = tpu::TPUGenerateTensorDesc ( grad_input );
  }
  if ( output_mask[1] == true )
  {
    grad_weight_bias_desc = tpu::TPUGenerateTensorDesc ( grad_weight );
  }
  else if ( output_mask[2] == true )
  {
    grad_weight_bias_desc = tpu::TPUGenerateTensorDesc ( grad_bias );
  }
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  status = sgdnn_batchnorm_backward_cudnn (
           handle,
           BatchNorm_Spatial,
           &alpha_data_diff,
           &beta_data_diff,
           &alpha_param_diff,
           &beta_param_diff,
           input_desc,
           ADDR_IN_DEVICE ( input ),
           grad_output_desc,
           ADDR_IN_DEVICE ( grad_output ),
           grad_input_desc,
           output_mask[0] ? ADDR_IN_DEVICE ( grad_input ) : nullptr,
           grad_weight_bias_desc,
           weight.defined() ? ADDR_IN_DEVICE ( weight ) : nullptr,
           output_mask[1] ? ADDR_IN_DEVICE ( grad_weight ) : nullptr,
           output_mask[2] ? ADDR_IN_DEVICE ( grad_bias ) : nullptr,
           eps,
           save_mean.defined() ? ADDR_IN_DEVICE ( save_mean ) : nullptr,
           save_invstd.defined() ? ADDR_IN_DEVICE ( save_invstd ) : nullptr,
           output_mask[0],
           output_mask[1],
           output_mask[2] );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BATCHNORM_BACKWARD, timer.ElapsedUS() );
#endif
  ++count;
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm_backward", native_batch_norm_backward_tpu );
}
} // namespace at
