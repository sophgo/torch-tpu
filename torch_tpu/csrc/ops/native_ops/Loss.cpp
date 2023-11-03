#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace torch {
namespace autograd {
class CrossEntropyLossFunction : public torch::autograd::Function<CrossEntropyLossFunction>
{
public:
  static at::Tensor forward (
  AutogradContext *ctx, const at::Tensor &self, const at::Tensor &target,
  const c10::optional<at::Tensor> &weight_opt, int64_t reduction,
  int64_t ignore_index, double label_smoothing )
  {
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["ignore_index"] = ignore_index;
    ctx->saved_data["label_smoothing"] = label_smoothing;
    bool weight_has_value = weight_opt.has_value();
    ctx->saved_data["weight_has_value"] = weight_has_value;
    if ( !weight_has_value )
    {
      ctx->save_for_backward ( {self, target} );
    }
    else
    {
      ctx->save_for_backward ( {self, target, weight_opt.value() } );
    }
    // do the compute and get result
    CHECK_TENSOR_IN_DEVICE ( self );
    CHECK_TENSOR_IN_DEVICE ( target );
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor ( weight_opt );
    const Tensor & weight = *weight_maybe_owned;
    if ( weight.defined() ) {CHECK_TENSOR_IN_DEVICE ( weight );};
#if 0
    auto out_cpu = cross_entropy_loss ( self.cpu(), target.cpu(),
                                        c10::optional<at::Tensor> ( weight.defined() ? weight.cpu() : Tensor() ),
                                        reduction, ignore_index, label_smoothing );
    auto out = out_cpu.to ( self.device() );
#else
    TensorOptions out_option = TensorOptions ( self.device() ).dtype ( self.dtype() );
    Tensor out = torch::empty ( {}, out_option );
    TORCH_CHECK ( reduction == 1 || reduction == 2 );
    TORCH_CHECK ( !weight.defined() );
    TORCH_CHECK ( ignore_index < 0 );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnCrossEntropyLoss (
                         tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor ( self ),
                         tpu::TPUGenerateSgdnnTensor ( target ),
                         reduction - 1,
                         label_smoothing,
                         tpu::TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CROSS_ENTROPY_LOSS, timer.ElapsedUS() );
#endif
#endif
    return out;
  }

  static tensor_list backward ( AutogradContext *ctx, tensor_list grad_outputs )
  {
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto ignore_index = ctx->saved_data["ignore_index"].toInt();
    auto label_smoothing = ctx->saved_data["label_smoothing"].toDouble();
    auto weight_has_value = ctx->saved_data["weight_has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto target = saved[1];
    CHECK_TENSOR_IN_DEVICE ( input );
    CHECK_TENSOR_IN_DEVICE ( target );
    c10::optional<at::Tensor> weight = c10::nullopt;
    if ( weight_has_value )
    {
      weight.emplace ( saved[2] );
      CHECK_TENSOR_IN_DEVICE ( weight.value() );
    }
    CHECK_TENSOR_IN_DEVICE ( grad_outputs[0] );
#if 0
    auto target_onehot_cpu = one_hot ( target.cpu(), input.sizes() [input.dim() - 1 ] );
    auto softmax_out_cpu = softmax ( input.cpu(), -1 );
    at::Tensor grad_input_cpu = ( softmax_out_cpu - target_onehot_cpu ) / target.size ( 0 );
    auto grad_input = grad_input_cpu.to ( input.device() );
#else
    at::Tensor grad_input = torch::empty ( input.sizes(), input.options() );
    TORCH_CHECK ( reduction == 1 || reduction == 2 );
    TORCH_CHECK ( !weight_has_value );
    TORCH_CHECK ( ignore_index < 0 );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnCrossEntropyLossBackward (
                         tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor ( input ),
                         tpu::TPUGenerateSgdnnTensor ( target ),
                         tpu::TPUGenerateSgdnnTensor ( grad_outputs[0] ),
                         reduction - 1,
                         label_smoothing,
                         tpu::TPUGenerateSgdnnTensor ( grad_input ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CROSS_ENTROPY_LOSS_BACKWARD, timer.ElapsedUS() );
#endif
#endif
    return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
  }
};
} //namespace autograd
} //namespace torch

namespace at
{
at::Tensor cross_entropy_loss_tpu ( const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, double label_smoothing )
{
  c10::optional<at::Tensor> weight = c10::nullopt;
  if ( weight_opt.has_value() && weight_opt.value().defined() )
  {
    weight = weight_opt;
  }
  return torch::autograd::CrossEntropyLossFunction::apply ( self, target, weight, reduction, ignore_index, label_smoothing );
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cross_entropy_loss", cross_entropy_loss_tpu );
}
TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
{
  m.impl ( "cross_entropy_loss", cross_entropy_loss_tpu );
}
} // namespace at
