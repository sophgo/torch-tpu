#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <vector>
#include "TPUTorchUtils.h"


#include "common/config.h"
#ifdef USING_PPL
#include "Adam.h"
#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, func)  \
AT_DISPATCH_SWITCH(                   \
scalar_type, name,                    \
AT_DISPATCH_CASE(at::kFloat, func)    \
AT_DISPATCH_CASE(at::kHalf, func)     \
AT_DISPATCH_CASE(at::kBFloat16, func) \
AT_DISPATCH_CASE(at::kInt, func)      \
AT_DISPATCH_CASE(at::kShort, func)    \
AT_DISPATCH_CASE(at::kChar, func)     \
AT_DISPATCH_CASE(at::kByte, func))

template <typename scalar_t>
static void adambackward_async(
    uint64_t weight_out,
    uint64_t m_out,
    uint64_t v_out,
    uint64_t vmax_out,
    uint64_t grad_weight,
    uint64_t weight_in,
    uint64_t m_in,
    uint64_t v_in,
    uint64_t vmax_in,
    uint64_t t,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    bool amsgrad,
    bool maximize,
    int inner_size)
{
auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
    uint32_t tile_size) -> int {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return adambackward_fp32(
      stream,
#ifndef BACKEND_SG2260
      ppl_module,
#endif
      weight_out,
      m_out,
      v_out,
      vmax_out,
      grad_weight,
      weight_in,
      m_in,
      v_in,
      vmax_in,
      t,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      amsgrad,
      maximize,
      inner_size,
      tile_size);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
    return adambackward_fp16(
      stream,
#ifndef BACKEND_SG2260
      ppl_module,
#endif
      weight_out,
      m_out,
      v_out,
      vmax_out,
      grad_weight,
      weight_in,
      m_in,
      v_in,
      vmax_in,
      t,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      amsgrad,
      maximize,
      inner_size,
      tile_size);
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    return adambackward_bf16(
      stream,
#ifndef BACKEND_SG2260
      ppl_module,
#endif
      weight_out,
      m_out,
      v_out,
      vmax_out,
      grad_weight,
      weight_in,
      m_in,
      v_in,
      vmax_in,
      t,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      amsgrad,
      maximize,
      inner_size,
      tile_size);
}
  return -1;
};

auto stream = c10_tpu::getCurrentTPUStream();
tpuKernelModule_t ppl_module = getPplModule();
int tile_size = inner_size;

while (tile_size >= 1) {
  int ret = kernel(stream, ppl_module, tile_size);
  if (ret == 0) {
    return;
  } else {
    tile_size = tile_size / 2;
    continue;
  }
}

TORCH_CHECK(false, "adambackward_async failed!");
}
#endif
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
#ifdef USING_PPL
  if (usePPLKernels())
  {
    uint32_t inner_size = 1;
    for (const auto j : c10::irange(self[i].dim())) {
        inner_size *= self[i].size(j);
    }
    AT_DISPATCH_FLOAT_INT_TYPES( self[i].scalar_type(), "adambackward_async", [&] {
            adambackward_async<scalar_t>(
                reinterpret_cast<uint64_t>(self[i].data_ptr()),
                reinterpret_cast<uint64_t>(exp_avgs[i].data_ptr()),
                reinterpret_cast<uint64_t>(exp_avg_sqs[i].data_ptr()),
                amsgrad ? reinterpret_cast<uint64_t>(max_exp_avg_sqs[i].data_ptr()) : 0ULL,
                reinterpret_cast<uint64_t>(grads[i].data_ptr()),
                reinterpret_cast<uint64_t>(self[i].data_ptr()),
                reinterpret_cast<uint64_t>(exp_avgs[i].data_ptr()),
                reinterpret_cast<uint64_t>(exp_avg_sqs[i].data_ptr()),
                amsgrad ? reinterpret_cast<uint64_t>(max_exp_avg_sqs[i].data_ptr()) : 0ULL,
                reinterpret_cast<uint64_t>(state_steps[i].data_ptr()),
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
                maximize,
                inner_size);
        });
    } else
#endif
    {
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
    }
    TIMING_END;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("_fused_adam_", _fused_adam_out_tpu);
}

void _fused_adamw_out_tpu(
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
        auto status = tpudnnAdamWBackwardMultiCoreAsync(
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
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS, "_fused_adamw_out_tpu failed.");
    }
    TIMING_END;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("_fused_adamw_", _fused_adamw_out_tpu);
}
}  // namespace at
