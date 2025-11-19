#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include "TPUTorchUtils.h"

#include <vector>
#include <algorithm>
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
    TIMING_START;
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

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnCrossEntropyLossAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, target),
        reduction - 1,
        ignore_index,
        label_smoothing,
        tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
    TIMING_END;
    SHOW_TENSOR_OP(self, target, out);
    return out;
  }

  static tensor_list backward ( AutogradContext *ctx, tensor_list grad_outputs )
  {
    TIMING_START;
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
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnCrossEntropyLossBackwardAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, input),
        tpu::TPUGenerateTpudnnTensor(stream, target),
        tpu::TPUGenerateTpudnnTensor(stream, grad_outputs[0]),
        ignore_index,
        reduction - 1,
        label_smoothing,
        tpu::TPUGenerateTpudnnTensor(stream, grad_input));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
    TIMING_END;
    SHOW_TENSOR_OP(input, target, grad_outputs[0], grad_input);
    return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
  }
};
} //namespace autograd
} //namespace torch

namespace at
{
Tensor cross_entropy_loss_tpu ( const Tensor& self, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, double label_smoothing )
{
  c10::optional<Tensor> weight = c10::nullopt;
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

//////////// **** binary_cross_entropy_with_logits **** ////////////
namespace at{
Tensor binary_cross_entropy_with_logits_tpu(const Tensor& input, const Tensor& target, 
  const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& pos_weight_opt, int64_t reduction) {
  TIMING_START;
#if 0
  std::optional<Tensor> weight_cpu = std::nullopt, pos_weight_cpu = std::nullopt;
  if ( weight.has_value() && weight.value().defined() )         weight_cpu = std::optional<Tensor> (weight.value().cpu());
  if ( pos_weight.has_value() && pos_weight.value().defined() ) pos_weight_cpu = std::optional<Tensor>(pos_weight.value().cpu());
  auto self_cpu   = self.cpu();
  auto target_cpu = target.cpu();
  auto out_cpu = torch::binary_cross_entropy_with_logits( self_cpu, target_cpu, weight_cpu, pos_weight_cpu, reduction );
  auto out     = out_cpu.to(self.device());
#endif
  auto log_sigmoid_input = at::log_sigmoid(input); // todo: imple logsigmoid

  if (pos_weight_opt.has_value() && pos_weight_opt->defined()) {
      // pos_weight need to be broadcasted, thus mul(target) is not inplace.
      auto log_weight = (*pos_weight_opt- 1).mul(target).add_(1);
      log_sigmoid_input.mul_(log_weight);
  }

  Tensor loss = (1 - target).mul_(input).sub_(log_sigmoid_input);

  if (weight_opt.has_value() && weight_opt->defined()) {
      loss.mul_(*weight_opt);
  }
  Tensor out;
  if (reduction == at::Reduction::Mean) {
      out = loss.mean();
  } else if (reduction == at::Reduction::Sum) {
      out = loss.sum();
  } else {
      out = loss;
  }
  TIMING_END;
  return out;
}
static bool isDefined(const std::optional<Tensor>& t) {
  return t.has_value() && t->defined();
}

Tensor binary_cross_entropy_with_logits_backward_tpu(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& pos_weight,
    int64_t reduction)
{
  TIMING_START;
  // Trivial case
  if (grad._is_zerotensor()) {
    return at::_efficientzerotensor(input.sizes(), input.options());
  }

  // -w * [ pos * y * (1 -sigmoid(x)) - (1 - y) sigmoid(x)] * grad

  // If there are subclassed tensors use the out of place version
  Tensor grad_input;
  if (isDefined(pos_weight)) {
    // pos_weight might need to be broadcasted, thus mul(target) is not inplace.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto t = pos_weight->mul(target);
    grad_input = at::areAnyTensorSubclassLike({input, target}) ||
            at::GradMode::is_enabled()
        ? t.add(1).sub(target).mul(input.sigmoid()).sub(t)
        : t.add(1).sub_(target).mul_(input.sigmoid()).sub_(t);
  } else {
    grad_input = at::areAnyTensorSubclassLike({input, target}) ||
            at::GradMode::is_enabled()
        ? input.sigmoid().sub(target)
        : input.sigmoid().sub_(target);
  }

  if (at::isTensorSubclassLike(grad) || at::GradMode::is_enabled()) {
    grad_input = grad_input.mul(grad);
  } else {
    grad_input.mul_(grad);
  }

  if (isDefined(weight)) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (at::isTensorSubclassLike(*weight) || at::GradMode::is_enabled()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      grad_input = grad_input.mul(*weight);
    } else {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      grad_input.mul_(*weight);
    }
  }

  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.sym_numel());
  }
  TIMING_END;

  return grad_input;
}

} // namespace at

namespace torch {
namespace autograd {
class BinaryCrossEntropyWithLogitsFunction : public torch::autograd::Function<BinaryCrossEntropyWithLogitsFunction>
{
public:
  static at::Tensor forward ( AutogradContext *ctx, const at::Tensor & self, const at::Tensor & target, 
    const ::std::optional<at::Tensor> & weight_opt, const ::std::optional<at::Tensor> & pos_weight_opt, int64_t reduction )
  {
    bool weight_has_value = weight_opt.has_value();
    bool pos_weight_has_value = pos_weight_opt.has_value();
    ctx->saved_data["weight_has_value"]     = weight_has_value;
    ctx->saved_data["pos_weight_has_value"] = pos_weight_has_value;
    ctx->saved_data["reduction"]            = reduction;
    auto weight     = weight_has_value ?      weight_opt.value() : at::Tensor();
    auto pos_weight = pos_weight_has_value ?  pos_weight_opt.value() : at::Tensor();
    ctx->save_for_backward ( {self, target, weight, pos_weight} );

    return at::binary_cross_entropy_with_logits_tpu(self, target, weight_opt, pos_weight_opt, reduction);
  }

  static tensor_list backward ( AutogradContext *ctx, tensor_list grad_outputs )
  {
    int64_t reduction         = ctx->saved_data["reduction"].toInt();
    bool weight_has_value     = ctx->saved_data["weight_has_value"].toBool();
    bool pos_weight_has_value = ctx->saved_data["pos_weight_has_value"].toBool();
    auto grad       = grad_outputs[0];
    auto saved      = ctx->get_saved_variables();
    auto input      = saved[0];
    auto target     = saved[1];
    auto weight     = saved[2];
    auto pos_weight = saved[3];
    std::optional<at::Tensor> weight_opt = std::nullopt, pos_weight_opt = std::nullopt;
    if ( weight_has_value )       weight_opt.emplace( weight );
    if ( pos_weight_has_value )   pos_weight_opt.emplace( pos_weight );

    auto inp_grad = at::binary_cross_entropy_with_logits_backward_tpu( grad, input, target, weight_opt, pos_weight_opt, reduction );
    return { inp_grad, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
  }
};
} //namespace autograd
} //namespace torch


namespace at{

Tensor binary_cross_entropy_with_logits_autogradtpu(const Tensor & self, const Tensor & target, 
  const ::std::optional<Tensor> & weight_opt, const ::std::optional<Tensor> & pos_weight_opt, int64_t reduction)
{
  std::optional<Tensor> weight = ::std::nullopt, pos_weight = ::std::nullopt;
  if ( weight_opt.has_value() && weight_opt.value().defined() ) { weight = weight_opt; }
  if ( pos_weight_opt.has_value() && pos_weight_opt.value().defined() ) { pos_weight = pos_weight_opt; }

  return torch::autograd::BinaryCrossEntropyWithLogitsFunction::apply( self, target, weight, pos_weight, reduction );
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_tpu );
}

TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
{
  m.impl ( "binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_autogradtpu );
}


}