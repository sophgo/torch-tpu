#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.trace.html#torch.trace
namespace at {
Tensor & trace_out_tpu(const Tensor & self, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::trace(self.cpu());
    out.copy_(out_cpu);
    return out;
}
Tensor trace_tpu(const Tensor & self) {
    auto out = empty({}, self.options());
    return trace_out_tpu(self, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("trace.out", trace_out_tpu);
    m.impl("trace", trace_tpu);
}
  
}