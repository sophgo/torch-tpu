#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/cpu/mixed_data_type.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

namespace at {

template <typename T>
void check_group_norm_inputs(const Tensor &input, const Tensor &weight,
                             const Tensor &bias, T C, int64_t num_groups) {
  TORCH_CHECK(num_groups > 0, "Expected num groups to be greater than 0, got ",
              num_groups);
  TORCH_CHECK(C % num_groups == 0,
              "Expected number of channels in input to be divisible by ",
              "num_groups, but got input of shape ", input.sizes(),
              " and "
              "num_groups=",
              num_groups);
  TORCH_CHECK(!weight.defined() ||
                  (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
              "Expected weight to be a vector of size equal to the number of ",
              "channels in input, but got weight of shape ", weight.sizes(),
              " and input of shape ", input.sizes());
  TORCH_CHECK(!bias.defined() ||
                  (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
              "Expected bias to be a vector of size equal to the number of ",
              "channels in input, but got bias of shape ", weight.sizes(),
              " and input of shape ", input.sizes());
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_tpu(
    const Tensor &X, const c10::optional<Tensor> &gamma_opt /* optional */,
    const c10::optional<Tensor> &beta_opt /* optional */, int64_t N, int64_t C,
    int64_t HxW, int64_t group, double eps) {
  CHECK_TENSOR_IN_DEVICE(X);
  const Tensor X_32 = X.to(caffe2::TypeMeta::Make<float>());
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor &gamma = *gamma_maybe_owned;
  const Tensor &beta = c10::value_or_else(beta_opt, [] { return Tensor(); });

  // nodechip need eps > 0
  if (eps <= 0)
    eps = 2.22507e-32;

  Tensor Y = empty_like(X_32);
  Tensor mean = at::empty({N, group}, X_32.options());
  Tensor rstd = at::empty({N, group}, X_32.options());

  auto weight = at::empty(gamma.sizes(), X_32.options());
  if (gamma_opt.has_value()) {
    // if gamma_opt has value, copy gamma_opt value to weight
    weight = weight.copy_(gamma_opt.value());
  }

  auto bias = at::empty(beta.sizes(), X_32.options());
  if (beta_opt.has_value()) {
    // if beta_opt has value, copy beta_opt value to bias
    bias = bias.copy_(beta_opt.value());
  }

  // set affine depend on gamma_opt and beta_opt
  int affine = 0;
  if (gamma_opt.has_value() && beta_opt.has_value()) {
    affine = 3;
  } else if (gamma_opt.has_value()) {
    affine = 1;
  } else if (beta_opt.has_value()) {
    affine = 2;
  } else {
    affine = 0;
  }

#if 0
  auto result =
      native_group_norm(X.cpu(), gamma_opt, beta_opt, N, C, HxW, group, eps);
  Y = std::get<0>(result).cpu();
  mean = std::get<1>(result).cpu();
  rstd = std::get<2>(result).cpu();
#else
  TIMING_START;
  #if defined BACKEND_1684X
  auto status = sgdnnNativeGroupNorm(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(X_32),
      tpu::TPUGenerateSgdnnTensor(weight), tpu::TPUGenerateSgdnnTensor(bias),
      group, affine, eps, tpu::TPUGenerateSgdnnTensor(Y),
      tpu::TPUGenerateSgdnnTensor(mean), tpu::TPUGenerateSgdnnTensor(rstd));
  TORCH_CHECK(status == BM_SUCCESS);
  #elif defined BACKEND_SG2260
  auto status = sgdnnNativeGroupNorm(
      c10_tpu::getCurrentTPUStream(), tpu::TPUGenerateSgdnnTensor(X_32),
      tpu::TPUGenerateSgdnnTensor(weight), tpu::TPUGenerateSgdnnTensor(bias),
      group, affine, eps, tpu::TPUGenerateSgdnnTensor(Y),
      tpu::TPUGenerateSgdnnTensor(mean), tpu::TPUGenerateSgdnnTensor(rstd));
  TORCH_CHECK(status == tpuRtSuccess);
  #endif
  TIMING_END(tpu::NATIVE_GROUP_NORM);
#endif
  return std::make_tuple(Y, mean, rstd);
}

// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("native_group_norm", native_group_norm_tpu);
// }

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_group_norm_backward_tpu(const at::Tensor &grad_out, const at::Tensor &X,
                               const at::Tensor &mean, const at::Tensor &rstd,
                               const c10::optional<at::Tensor> &weight_opt,
                               int64_t N, int64_t C, int64_t HxW, int64_t group,
                               ::std::array<bool, 3> output_mask) {
  CHECK_TENSOR_IN_DEVICE(X);
  CHECK_TENSOR_IN_DEVICE(grad_out);
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor &weight = *weight_maybe_owned;
  TORCH_CHECK(weight.defined(), "weight must be defined");
  if (weight.defined()) {
    CHECK_TENSOR_IN_DEVICE(weight);
  }
#if 0
    // tpu impl has bug. to fix
    CPU_IMPL_WARNING();
    TIMING_START;
    auto input_type = grad_out.dtype();
    auto result = native_group_norm_backward(
        grad_out.cpu().to(torch::kFloat32), X.cpu().to(torch::kFloat32),
        mean.cpu().to(torch::kFloat32), rstd.cpu().to(torch::kFloat32),
        c10::optional<Tensor>(weight.cpu().to(torch::kFloat)), N, C, HxW,
        group, output_mask);
    TIMING_END(tpu::CPU_LAYER);
    return std::tuple<Tensor, Tensor, Tensor>(
        output_mask[0] ? TENSOR_TO_TPU(std::get<0>(result).to(input_type).contiguous())
                      : Tensor(),
        output_mask[1] ? TENSOR_TO_TPU(std::get<1>(result).to(input_type).contiguous())
                      : Tensor(),
        output_mask[2] ? TENSOR_TO_TPU(std::get<2>(result).to(input_type).contiguous())
                      : Tensor());
#else
  TORCH_CHECK(X.dim() == 4);
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0] == true) {
    grad_input = torch::empty(X.sizes(), X.options());
  }
  if (output_mask[1] == true) {
    grad_weight = empty(weight.sizes(), weight.options());
  }
  if (output_mask[2] == true) {
    // We assume that weight and bias have the same data type
    grad_bias = empty({weight.size(0)}, weight.options());
  }

  auto mean1 = mean.view({N, group, 1});
  auto rstd1 = rstd.view({N, group, 1});
  auto X1    = X.view({N,group, C/group * HxW});
  auto x_hat = (X1 - mean1) * rstd1;
  if (output_mask[0] == true) { // grad inp
    auto dx_hat = (grad_out * weight.view({1, C, 1, 1})).view({N, group, C/group * HxW});
    grad_input = ((dx_hat - dx_hat.mean(-1, true) - (dx_hat * x_hat).mean(-1, true) * x_hat) * rstd1).view(X.sizes());
  }
  if (output_mask[1] == true) {
    grad_weight = (grad_out.view({N, group, C/group * HxW}) * x_hat).view({N, C, -1}).sum(0).sum(-1);
  }
  if (output_mask[2] == true) {
    grad_bias = grad_out.sum(0).sum(-1).sum(-1);
  }
  // TODO:FIX BUG
  // TIMING_START;
  // auto status = sgdnnNativeGroupNormBackward(
  //     tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(grad_out),
  //     tpu::TPUGenerateSgdnnTensor(X),
  //     weight.defined() ? tpu::TPUGenerateSgdnnTensor(weight)
  //                      : sgdnnUndefinedTensor(),
  //     tpu::TPUGenerateSgdnnTensor(mean), tpu::TPUGenerateSgdnnTensor(rstd),
  //     group,
  //     output_mask[0] ? tpu::TPUGenerateSgdnnTensor(grad_input)
  //                    : sgdnnUndefinedTensor(),
  //     output_mask[1] ? tpu::TPUGenerateSgdnnTensor(grad_weight)
  //                    : sgdnnUndefinedTensor(),
  //     output_mask[1] ? tpu::TPUGenerateSgdnnTensor(grad_bias)
  //                    : sgdnnUndefinedTensor());
  // TORCH_CHECK(status == BM_SUCCESS);
  // TIMING_END(tpu::GROUPNORM_BACKWARD)
  // SHOW_TENSOR_OP(grad_out, X, mean, rstd, weight, grad_input, grad_weight, grad_bias);
  return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);

#endif
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("native_group_norm_backward", native_group_norm_backward_tpu);
}

} // namespace at
