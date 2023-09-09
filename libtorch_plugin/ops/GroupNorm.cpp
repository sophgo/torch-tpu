#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/cpu/moments_utils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>
// #include "tpu_kernel.h"

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
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnNativeGroupNorm(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(X_32),
      tpu::TPUGenerateSgdnnTensor(weight), tpu::TPUGenerateSgdnnTensor(bias),
      group, affine, eps, tpu::TPUGenerateSgdnnTensor(Y),
      tpu::TPUGenerateSgdnnTensor(mean), tpu::TPUGenerateSgdnnTensor(rstd));
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::NATIVE_GROUP_NORM, timer.ElapsedUS());
#endif
#endif
  return std::make_tuple(Y, mean, rstd);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("native_group_norm", native_group_norm_tpu);
}
} // namespace at
