#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.renorm.html#torch.renorm
namespace at {
Tensor & renorm_out_tpu(const Tensor & self, const Scalar & p, int64_t dim, const Scalar & maxnorm, Tensor & out){
    CPU_IMPL_WARNING();
    auto out_cpu = torch::renorm(self.cpu(), p, dim, maxnorm);
    out = out_cpu.to(out.device());
    return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("renorm.out", renorm_out_tpu);
}
}