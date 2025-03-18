#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
at::Tensor & linspace_out_tpu(const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out)
{
    CPU_IMPL_WARNING();
    auto out_cpu = linspace( start, end, steps );
    out = out_cpu.to(out.device()).to(out.dtype());
    return out;
}


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
    m.impl ( "linspace.out",         linspace_out_tpu);
}
} // namespace at