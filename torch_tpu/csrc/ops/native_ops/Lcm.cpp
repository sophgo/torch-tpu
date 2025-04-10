#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.lcm.html#torch.lcm
namespace at {
Tensor & lcm_out_tpu(const Tensor & self, const Tensor & other, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::lcm(self.cpu(), other.cpu());
    out = out_cpu.to(out.device());
    return out;
}
Tensor & lcm__tpu(Tensor & self, const Tensor & other) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::lcm(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice ( self.data_ptr(), out_cpu.contiguous().data_ptr(), self.nbytes() );
    return self;
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("lcm.out", lcm_out_tpu);
    m.impl("lcm_",    lcm__tpu);
}
}