#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <vector>
#include "TPUTorchUtils.h"


#include "common/config.h"
namespace at {
void _fused_adam_out_tpu(
    at::TensorList self,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
)
{ 
    TIMING_START;
    // Check that all tensors are on the same device
    for (const auto& s : self) {
        CHECK_TENSOR_IN_DEVICE(s);
    }
    for (const auto& g : grads) {
        CHECK_TENSOR_IN_DEVICE(g);
    }
    for (const auto& e : exp_avgs) {
        CHECK_TENSOR_IN_DEVICE(e);
    }
    for (const auto& v : exp_avg_sqs) {
        CHECK_TENSOR_IN_DEVICE(v);
    }
    if(amsgrad){
        for (const auto& m : max_exp_avg_sqs) {
            CHECK_TENSOR_IN_DEVICE(m);
        }
    }
    for (const auto& s : state_steps) {
        CHECK_TENSOR_IN_DEVICE(s);
    }
    // Check that all tensors are on the same shape
    TORCH_CHECK(self.size() == grads.size() &&
                self.size() == exp_avgs.size() &&
                self.size() == exp_avg_sqs.size(), "All TensorLists must have the same size.");
    if(amsgrad){
        TORCH_CHECK(
            self.size() == max_exp_avg_sqs.size(), "Parameter 'max_exp_avg_sqs' must have the same size with other TensorLists parameters."
        )
    }

    // Get the current TPU stream
    auto stream = c10_tpu::getCurrentTPUStream();
    // Adam need inplace operation
    for (size_t i = 0; i < self.size(); ++i) {
        auto status = tpudnnAdamBackwardMultiCoreAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, self[i]),
            tpu::TPUGenerateTpudnnTensor(stream, exp_avgs[i]),
            tpu::TPUGenerateTpudnnTensor(stream, exp_avg_sqs[i]),
            amsgrad ? tpu::TPUGenerateTpudnnTensor(stream, max_exp_avg_sqs[i]) : tpudnnUndefinedTensor(),
            tpu::TPUGenerateTpudnnTensor(stream, grads[i]),
            tpu::TPUGenerateTpudnnTensor(stream, self[i]),
            tpu::TPUGenerateTpudnnTensor(stream, exp_avgs[i]),
            tpu::TPUGenerateTpudnnTensor(stream, exp_avg_sqs[i]),
            amsgrad ? tpu::TPUGenerateTpudnnTensor(stream, max_exp_avg_sqs[i]) : tpudnnUndefinedTensor(),
            tpu::TPUGenerateTpudnnTensor(stream, state_steps[i]),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            maximize
        );
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS, "_fused_adam_out_tpu failed.");
    }
    TIMING_END;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("_fused_adam_", _fused_adam_out_tpu);
}
}  // namespace at
