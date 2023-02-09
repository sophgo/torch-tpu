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
const Tensor & input,
const c10::optional<Tensor> & weight,
const c10::optional<Tensor> & bias,
const c10::optional<Tensor> & running_mean,
const c10::optional<Tensor> & running_var,
bool training,
double momentum,
double eps )
{
  LOG ( FATAL ) << "Bathnorm is unsupported by TPU";
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "native_batch_norm", native_batch_norm_tpu );
}
} // namespace at
