#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.cross.html#torch.cross
namespace at {
Tensor & linalg_cross_out_tpu(const Tensor & self, const Tensor & other, int64_t dim, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::linalg_cross(self.cpu(), other.cpu(), dim);
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
    return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("linalg_cross.out", linalg_cross_out_tpu);
}
  
}