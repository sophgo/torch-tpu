#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.roll.html#torch.roll
namespace at {

Tensor & remainder_tensor_out_tpu(const Tensor & self, const Tensor & other, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::remainder(self.cpu(), other.cpu());
    out = TENSOR_TO_TPU ( out_cpu );
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