#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include <unistd.h>

#define TPU_LIBTORCH_OP_COMPARE
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
  //std::cout << "Batchnorm" << std::endl;
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
    TORCH_CHECK ( weight.is_contiguous() );
  }
  if ( bias.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( bias );
    TORCH_CHECK ( bias.is_contiguous() );
  }
  if ( running_mean.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( running_mean );
    TORCH_CHECK ( running_mean.is_contiguous() );
  }
  if ( running_var.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( running_var );
    TORCH_CHECK ( running_var.is_contiguous() );
  }
#if 1
  auto outputs_cpu = native_batch_norm (
                     input.to ( torch::Device ( "cpu" ) ),
                     c10::optional<Tensor> ( weight.defined() ? weight.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( bias.defined() ? bias.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( running_mean.defined() ? running_mean.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( running_var.defined() ? running_var.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     training,
                     momentum,
                     eps );
  if ( running_mean.defined() )
  {
    //const_cast<Tensor &> ( running_mean ) = running_mean_cpu.to ( tpu::TPUGetCurrentDevice() );
  }
  if ( running_var.defined() )
  {
    //const_cast<Tensor &> ( running_var ) = running_var_cpu.to ( tpu::TPUGetCurrentDevice() );
  }
#if 0
  auto output2 = std::get<2> ( outputs_cpu ).contiguous().to ( tpu::TPUGetCurrentDevice() );
  auto output1 = std::get<1> ( outputs_cpu ).contiguous().to ( tpu::TPUGetCurrentDevice() );
  auto output0 = std::get<0> ( outputs_cpu ).contiguous().to ( tpu::TPUGetCurrentDevice() );
#else
  auto output0 = std::get<0> ( outputs_cpu ).contiguous().to ( tpu::TPUGetCurrentDevice() );
  auto output1 = std::get<1> ( outputs_cpu ).contiguous().to ( tpu::TPUGetCurrentDevice() );
  auto output2 = std::get<2> ( outputs_cpu ).contiguous().to ( tpu::TPUGetCurrentDevice() );
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( output0, output1, output2 );
#else
#ifndef TPU_LIBTORCH_OP_COMPARE
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
  auto buffer = torch::empty ( input.sizes(), output_options );
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
           ADDR_IN_DEVICE ( save_invstd ),
           ADDR_IN_DEVICE ( buffer ));
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BATCHNORM, timer.ElapsedUS() );
#endif
#ifndef TPU_LIBTORCH_OP_COMPARE
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
  std::cout << "Compare output" << std::endl;
  auto output_got = output.to ( torch::Device ( "cpu" ) );
  auto output_exp = std::get<0> ( outputs_exp );
  tpu::TPUCompareResult ( output_got, output_exp, 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) )  );
  std::cout << "Compare save_mean" << std::endl;
  auto save_mean_got = save_mean.to ( torch::Device ( "cpu" ) );
  auto save_mean_exp = std::get<1> ( outputs_exp );
  tpu::TPUCompareResult ( save_mean_got, save_mean_exp, 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) )  );
  std::cout << "Compare save_invstd" << std::endl;
  auto save_invstd_got = save_invstd.to ( torch::Device ( "cpu" ) );
  auto save_invstd_exp = std::get<2> ( outputs_exp );
  tpu::TPUCompareResult ( save_invstd_got, save_invstd_exp, 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) )  );
  if ( running_mean.defined() )
  {
    auto running_mean_got = running_mean.to ( torch::Device ( "cpu" ) );
    std::cout << "Compare running_mean" << std::endl;
    tpu::TPUCompareResult ( running_mean_got, running_mean_cpu, 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) ) );
  }
  if ( running_var.defined() )
  {
    auto running_var_got = running_var.to ( torch::Device ( "cpu" ) );
    std::cout << "Compare running_mean" << std::endl;
    tpu::TPUCompareResult ( running_var_got, running_var_cpu, 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) ) );
  }
#endif
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
  TORCH_CHECK ( grad_output.is_contiguous() );
  CHECK_TENSOR_IN_DEVICE ( input );
  TORCH_CHECK ( input.is_contiguous() );
  if ( weight.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( weight );
    TORCH_CHECK ( weight.is_contiguous() );
  }
  if ( save_mean.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( save_mean );
    TORCH_CHECK ( save_mean.is_contiguous() );
  }
  if ( save_invstd.defined() )
  {
    CHECK_TENSOR_IN_DEVICE ( save_invstd );
    TORCH_CHECK ( save_invstd.is_contiguous() );
  }
#if 0
  auto outputs_cpu = native_batch_norm_backward (
                     grad_output.to ( torch::Device ( "cpu" ) ),
                     input.to ( torch::Device ( "cpu" ) ),
                     c10::optional<Tensor> ( weight.defined() ? weight.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( running_mean.defined() ? running_mean.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( running_var.defined() ? running_var.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( save_mean.defined() ? save_mean.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     c10::optional<Tensor> ( save_invstd.defined() ? save_invstd.to ( torch::Device ( "cpu" ) ) : Tensor() ),
                     training,
                     eps,
                     output_mask );
  return std::tuple<Tensor, Tensor, Tensor> (
         output_mask[0] ? std::get<0> ( outputs_cpu ).to ( tpu::TPUGetCurrentDevice() ) : Tensor(),
         output_mask[1] ? std::get<1> ( outputs_cpu ).to ( tpu::TPUGetCurrentDevice() ) : Tensor(),
         output_mask[2] ? std::get<2> ( outputs_cpu ).to ( tpu::TPUGetCurrentDevice() ) : Tensor() );
#else
#ifndef TPU_LIBTORCH_OP_COMPARE
  auto grad_output_cpu = grad_output.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  at::Tensor weight_cpu, running_mean_cpu, running_var_cpu, save_mean_cpu, save_invstd_cpu;
  if ( weight.defined() )
  {
    weight_cpu = weight.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  }
  if ( running_mean.defined() )
  {
    running_mean_cpu = running_mean.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  }
  if ( running_var.defined() )
  {
    running_var_cpu = running_var.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  }
  if ( save_mean.defined() )
  {
    save_mean_cpu = save_mean.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  }
  if ( save_invstd.defined() )
  {
    save_invstd_cpu = save_invstd.to ( torch::Device ( "cpu" ) ).to ( torch::kDouble );
  }
  auto outputs_exp = native_batch_norm_backward (
                     grad_output_cpu,
                     input_cpu,
                     c10::optional<Tensor> ( weight_cpu ),
                     c10::optional<Tensor> ( running_mean_cpu ),
                     c10::optional<Tensor> ( running_var_cpu ),
                     c10::optional<Tensor> ( save_mean_cpu ),
                     c10::optional<Tensor> ( save_invstd_cpu ),
                     training,
                     eps,
                     output_mask );
#endif
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
#ifndef TPU_LIBTORCH_OP_COMPARE
  std::cout << "Comparing batchnorm backward:"
            << " grad_output shape = " << grad_output.sizes()
            << " grad_output dtype = " << grad_output.dtype()
            << " input shape = " << input.sizes()
            << " input dtype = " << input.dtype()
            << " weight shape = " << weight.sizes()
            << " weight dtype = " << weight.dtype()
            << " save_mean shape = " << save_mean.sizes()
            << " save_mean dtype = " << save_mean.dtype()
            << " save_invstd shape = " << save_invstd.sizes()
            << " save_invstd dtype = " << save_invstd.dtype()
            << " grad_input shape = " << grad_input.sizes()
            << " grad_input dtype = " << grad_input.dtype()
            << " grad_weight shape = " << grad_weight.sizes()
            << " grad_weight dtype = " << grad_weight.dtype()
            << " grad_bias shape = " << grad_bias.sizes()
            << " grad_bias dtype = " << grad_bias.dtype()
            << std::endl;
  if ( output_mask[0] == true )
  {
    std::cout << "Compare grad_input" << std::endl;
    auto grad_input_got = grad_input.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( grad_input_got, std::get<0> ( outputs_exp ).to ( torch::kFloat ), 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) ) );
  }
  if ( output_mask[1] == true )
  {
    std::cout << "Compare grad_weight" << std::endl;
    auto grad_weight_got = grad_weight.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( grad_weight_got, std::get<1> ( outputs_exp ).to ( torch::kFloat ), 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) ) );
  }
  if ( output_mask[2] == true )
  {
    std::cout << "Compare grad_bias" << std::endl;
    auto grad_bias_got = grad_bias.to ( torch::Device ( "cpu" ) );
    tpu::TPUCompareResult ( grad_bias_got, std::get<2> ( outputs_exp ).to ( torch::kFloat ), 1.0 / ( ( double ) input.size ( 0 ) * input.size ( 2 ) * input.size ( 3 ) ) );
  }
#endif
  return std::tuple<Tensor, Tensor, Tensor> ( grad_input, grad_weight, grad_bias );
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm_backward", native_batch_norm_backward_tpu );
}
} // namespace at
