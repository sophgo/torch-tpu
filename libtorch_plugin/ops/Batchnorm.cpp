#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

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
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
  const Tensor & weight = *weight_maybe_owned;
  const Tensor & bias = c10::value_or_else ( bias_opt, [] {return Tensor();} );
  const Tensor & running_mean = c10::value_or_else ( running_mean_opt, [] {return Tensor();} );
  const Tensor & running_var = c10::value_or_else ( running_var_opt, [] {return Tensor();} );
  auto num_features = input.size ( 1 );
  LOG ( FATAL ) << "Bathnorm is unsupported by TPU";
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm", native_batch_norm_tpu );
}
} // namespace at
