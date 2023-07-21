#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"
namespace torch{
namespace autograd{
class DropOutFunction : public torch::autograd::Function<DropOutFunction>
{
public:
  static at::Tensor forward(AutogradContext *ctx, const at::Tensor& self,
                            double p, bool train)
  {
  TensorOptions option = TensorOptions( ).dtype ( self.dtype() );
  at::Tensor mask_cpu = torch::empty( self.sizes(), option );
  at::Tensor mask = torch::bernoulli(mask_cpu, p).to(self.device());
  ctx->saved_data["p"] = p;
  ctx->saved_data["train"] = train;
  ctx->save_for_backward( {mask} );
  auto out = mask * self * (1/(1-p));
  return out;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list gradout)
  {
    auto p = ctx->saved_data["p"].toDouble();
    auto train = ctx->saved_data["train"].toBool();
    auto saved = ctx->get_saved_variables();
    auto mask = saved[0];
    auto gradinp = mask * gradout[0];
    return {gradinp, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
  }
};
} // namespace autograd
} // namespace torch


namespace at
{
Tensor dropout_tpu(const at::Tensor & input, double p, bool train) {  
  return torch::autograd::DropOutFunction::apply(input, p, train);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "dropout", dropout_tpu );
}
TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
{
  m.impl ( "dropout", dropout_tpu );
}
}//namespace at

namespace at
{
Tensor & dropout__tpu(at::Tensor & input, double p, bool train) {
#if 1
  auto input_cpu =input.cpu();
  auto out_cpu = dropout_(input_cpu, p, train);
  input = out_cpu.to(input.device());
#endif
  return input;
}

// TORCH_LIBRARY_IMPL ( aten, TPU, m )
// {
//   m.impl ( "dropout_", dropout__tpu );
// }
// TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
// {
//   m.impl ( "dropout_", dropout__tpu );
// }

} // namespace at