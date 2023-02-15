#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_LIBTORCH_OP_COMPARE TRUE

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
  TORCH_CHECK ( weight.defined(), "weight must be defined" );
  TORCH_CHECK ( bias.defined(), "bias must be defined" );
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
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  at::Tensor weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu;
  if ( weight.defined() )
  {
    weight_cpu = weight.to ( torch::Device ( "cpu" ) );
  }
  if ( bias.defined() )
  {
    bias_cpu = bias.to ( torch::Device ( "cpu" ) );
  }
  if ( running_mean.defined() )
  {
    running_mean_cpu = running_mean.to ( torch::Device ( "cpu" ) );
  }
  if ( running_var.defined() )
  {
    running_var_cpu = running_var.to ( torch::Device ( "cpu" ) );
  }
  auto outputs_exp = native_batch_norm (
                     input_cpu,
                     c10::optional<Tensor> ( weight_cpu ),
                     c10::optional<Tensor> ( bias_cpu ),
                     c10::optional<Tensor> ( running_mean_cpu ),
                     c10::optional<Tensor> ( running_var_cpu ),
                     training,
                     momentum,
                     eps );
#endif
  auto output_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( input.dtype() );
  auto output = torch::empty ( input.sizes(), output_options );
  auto other_options = torch::TensorOptions ( tpu::TPUGetCurrentDevice() ).dtype ( weight.dtype() );
  auto save_mean = torch::empty_symint ( c10::SymIntArrayRef ( { num_features } ), other_options );
  auto save_invstd = torch::empty_symint ( c10::SymIntArrayRef ( { num_features } ), other_options );
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
  float alpha = 1.f;
  float beta = 0.f;
  auto input_desc = tpu::TPUGenerateTensorDesc ( input );
  auto output_desc = tpu::TPUGenerateTensorDesc ( output );
  auto other_desc = tpu::TPUGenerateTensorDesc ( weight );
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
#ifdef TPU_LIBTORCH_OP_COMPARE
  std::cout << "Comparing batchnorm:"
            << " input shape = " << input.sizes()
            << " input dtype = " << input.dtype()
            << " weight shape = " << weight.sizes()
            << " weight dtype = " << weight.dtype()
            << " bias shape = " << bias.sizes()
            << " bias dtype = " << bias.dtype()
            << " running_mean shape = " << running_mean.sizes()
            << " running_mean dtype = " << running_mean.dtype()
            << " running_var shape = " << running_var.sizes()
            << " running_var dtype = " << running_var.dtype()
            << " output shape = " << output.sizes()
            << " output dtype = " << output.dtype()
            << " save_mean shape = " << save_mean.sizes()
            << " save_mean dtype = " << save_mean.dtype()
            << " save_invstd shape = " << save_invstd.sizes()
            << " save_invstd dtype = " << save_invstd.dtype()
            << std::endl;
  std::cout << "Compare output\n";
  auto output_got = output.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( output_got, std::get<0> ( outputs_exp ) );
  std::cout << "Compare save_mean\n";
  auto save_mean_got = save_mean.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( save_mean_got, std::get<1> ( outputs_exp ) );
  std::cout << "Compare save_invstd\n";
  auto save_invstd_got = save_invstd.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( save_invstd_got, std::get<2> ( outputs_exp ) );
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( output, save_mean, save_invstd );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm", native_batch_norm_tpu );
}
} // namespace at
