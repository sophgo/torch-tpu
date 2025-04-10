#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.eye.html#torch.eye
namespace at {
Tensor & eye_m_out_tpu(int64_t n, int64_t m, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::eye(m, n);
    out = out_cpu.to(out.device());
    return out;
}
Tensor & eye_out_tpu(int64_t n, at::Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::eye(n);
    out = out_cpu.to(out.device());
    return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("eye.m_out", eye_m_out_tpu);
    m.impl("eye.out",   eye_out_tpu);
}
}