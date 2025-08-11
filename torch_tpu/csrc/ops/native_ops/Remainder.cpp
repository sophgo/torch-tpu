#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://docs.pytorch.org/docs/2.1/generated/torch.remainder.html#torch-remainder
namespace at {

Tensor & remainder_tensor_out_tpu(const Tensor & self, const Tensor & other, Tensor & out) {

    if ( other.dim() == 0 && IS_CPU_TENSOR(other))
    {
        auto self_ = self.contiguous(); if ( !self.is_contiguous() ) { CONTIGUOUS_WARNING(); }
        TIMING_START;
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnRemainderAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, self_),
            other.item().toFloat(),
            tpu::TPUGenerateTpudnnTensor(stream, out));
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        TIMING_END(Remainder)
    }
    else
    {
        CPU_IMPL_WARNING();
        auto out_cpu = torch::remainder(self.cpu(), other.cpu());
        out = TENSOR_TO_TPU ( out_cpu );
    }

    return out;
}
// Tensor remainder_tensor_tpu(const Tensor & self, const Tensor & other) {
// }

// Tensor & remainder_scalar_out_tpu(const Tensor & self, const Scalar & other, Tensor & out) {
// }

// Tensor remainder_scalar_tpu(const Tensor & self, const Scalar & other) {
// }

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("remainder.Tensor_out", remainder_tensor_out_tpu);
    // m.impl("remainder.Tensor",     remainder_tensor_tpu);
    // m.impl("remainder.Scalar_out", remainder_scalar_out_tpu);
    // m.impl("remainder.Scalar",     remainder_scalar_tpu);
}

}