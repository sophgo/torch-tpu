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
  CHECK_TENSOR_IN_DEVICE ( self );
  ctx->saved_data["p"] = p;
  ctx->saved_data["train"] = train;
  if ( p == 0 || !train ) {
    return self;
  } else {
#if 1
    TensorOptions option = TensorOptions( ).device("cpu").dtype ( self.dtype() );
    at::Tensor mask_cpu = torch::rand_like(self, option) > p;
    at::Tensor mask = mask_cpu.to(self.device()).to(self.dtype());
#endif
    ctx->save_for_backward( {mask} );
    auto out = mask * self * (1/(1-p));
    SHOW_TENSOR_OP(self, out);
    return out;
  }
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list gradout)
  {
    auto p = ctx->saved_data["p"].toDouble();
    auto train = ctx->saved_data["train"].toBool();
    if (p == 0 || !train) {
      return {gradout[0], at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
    }else{
      auto saved = ctx->get_saved_variables();
      auto mask = saved[0];
      auto gradinp = mask * gradout[0] * (1/(1-p));
      SHOW_TENSOR_OP(gradinp);
      return {gradinp, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
    }
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