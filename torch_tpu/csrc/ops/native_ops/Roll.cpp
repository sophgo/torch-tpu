#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.roll.html#torch.roll
namespace at {

Tensor & roll_out_tpu(const Tensor & self, IntArrayRef shifts, IntArrayRef dims, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::roll(self.cpu(), shifts, dims);
    out.copy_( out_cpu );
    return out;
}
Tensor roll_tpu(const Tensor & self, IntArrayRef shifts, IntArrayRef dims) {
    auto out = empty(self.sizes(), self.options());
    return roll_out_tpu(self, shifts, dims, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("roll.out", roll_out_tpu);
    m.impl("roll",     roll_tpu);
}

}