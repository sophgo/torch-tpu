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
    for (const auto& m : max_exp_avg_sqs) {
        CHECK_TENSOR_IN_DEVICE(m);
    }
    for (const auto& s : state_steps) {
        CHECK_TENSOR_IN_DEVICE(s);
    }
    // Check that all tensors are on the same shape
    TORCH_CHECK(self.size() == grads.size() &&
                self.size() == exp_avgs.size() &&
                self.size() == exp_avg_sqs.size() &&
                self.size() == max_exp_avg_sqs.size(),
                // self.size() == out.size(),
                "All TensorLists must have the same size.");
        // Get the current TPU stream
    auto stream = c10_tpu::getCurrentTPUStream();

    // Generate TPU-specific tensors
    std::vector<at::Tensor> output;
    std::vector<at::Tensor> output_m;
    std::vector<at::Tensor> output_v;
    std::vector<at::Tensor> output_vmax;

    // Initialize output vectors with empty tensors
    for (size_t i = 0; i < self.size(); ++i) {
        output.push_back(torch::empty(self[i].sizes(), self[i].options()));
        output_m.push_back(torch::empty(exp_avgs[i].sizes(), exp_avgs[i].options()));
        output_v.push_back(torch::empty(exp_avg_sqs[i].sizes(), exp_avg_sqs[i].options()));
        output_vmax.push_back(torch::empty(max_exp_avg_sqs[i].sizes(), max_exp_avg_sqs[i].options()));
    }

    TIMING_START;
    for (size_t i = 0; i < self.size(); ++i) {
        auto status = tpudnnAdamBackwardMultiCoreAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, output[i]),
            tpu::TPUGenerateTpudnnTensor(stream, output_m[i]),
            tpu::TPUGenerateTpudnnTensor(stream, output_v[i]),
            tpu::TPUGenerateTpudnnTensor(stream, output_vmax[i]),
            tpu::TPUGenerateTpudnnTensor(stream, grads[i]),
            tpu::TPUGenerateTpudnnTensor(stream, self[i]),
            tpu::TPUGenerateTpudnnTensor(stream, exp_avgs[i]),
            tpu::TPUGenerateTpudnnTensor(stream, exp_avg_sqs[i]),
            tpu::TPUGenerateTpudnnTensor(stream, max_exp_avg_sqs[i]),
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
    self = at::TensorList(output.data(), output.size());
    TIMING_END(tpu::ADAM_BACKWARD);
    //SHOW_TENSOR_OP(self);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("_fused_adam_", _fused_adam_out_tpu);
}
}  // namespace at
