#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
inline at::Tensor & copysign_out_tpu(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    TIMING_START;
    CPU_IMPL_WARNING();
    auto out_cpu = copysign( self.cpu(), other.cpu() );
    out = out_cpu.to(out.device()).to(out.dtype());
    TIMING_END;
    return out;
}

inline at::Tensor & copysign_Scalar_out_tpu(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)
{
    TIMING_START;
    CPU_IMPL_WARNING();
    auto out_cpu = copysign( self.cpu(), other );
    out = out_cpu.to(out.device()).to(out.dtype());
    TIMING_END;
    return out;
}

inline at::Tensor copysign_Scalar_tpu(const at::Tensor & self, const at::Scalar & other)
{
    auto out = empty(self.sizes(), self.options());
    return copysign_Scalar_out_tpu(self, other, out);
}

inline at::Tensor copysign_Tensor_tpu(const at::Tensor & self, const at::Tensor & other)
{
    auto out = empty(self.sizes(), self.options());
    return copysign_out_tpu(self, other, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
    m.impl ( "copysign.out",            copysign_out_tpu);
    m.impl ( "copysign.Scalar_out",     copysign_Scalar_out_tpu);
    m.impl ( "copysign.Scalar",         copysign_Scalar_tpu);
    m.impl ( "copysign.Tensor",         copysign_Tensor_tpu);
}
} // namespace at