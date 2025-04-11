#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.trace.html#torch.trace
namespace at {
Tensor & tril_out_tpu(const Tensor & self, int64_t diagonal, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::tril(self.cpu(), diagonal);
    out.copy_(out_cpu);
    return out;
}
Tensor tril_tpu(const Tensor & self, int64_t diagonal=0) {
    auto out = empty(self.sizes(), self.options());
    return tril_out_tpu(self, diagonal, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("tril.out", tril_out_tpu);
    m.impl("tril", tril_tpu);
}

// Tensor & tril_indices_out_tpu(int64_t row, int64_t col, int64_t offset, Tensor & out) {
//     CPU_IMPL_WARNING();
//     auto out_cpu = torch::tril_indices(row)
// }
// Tensor tril_indices_tpu(int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {

// }
// TORCH_LIBRARY_IMPL(aten, TPU, m)
// {
//     m.impl("tril_indices.out", tril_indices_out_tpu);
//     m.impl("tril_indices",     tril_indices_tpu);
// }


}