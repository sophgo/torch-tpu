#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>

#include <ATen/NativeFunctions.h>

#include "TPUTorchUtils.h"

namespace at {
Tensor & all_out_tpu(const Tensor & self, int64_t dim, bool keepdim, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = all(self.cpu(), dim, keepdim);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    return out;
}

Tensor & all_all_out(const at::Tensor & self, at::Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = all(self.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    return out;
}

// https://pytorch.org/docs/2.1/generated/torch.all.html
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("all.out", all_out_tpu);
    m.impl("all.all_out", all_all_out);
}
} // namespace at