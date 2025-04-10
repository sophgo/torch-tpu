#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.gcd.html#torch.gcd
namespace at {
Tensor & gcd_out_tpu(const Tensor & self, const Tensor & other, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::gcd(self.cpu(), other.cpu());
    out = out_cpu.to(out.device());
    return out;
}
Tensor & gcd__tpu(Tensor & self, const Tensor & other) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::gcd(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice ( self.data_ptr(), out_cpu.contiguous().data_ptr(), self.nbytes() );
    return self;
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("gcd.out", gcd_out_tpu);
    m.impl("gcd_",    gcd__tpu);
}
}