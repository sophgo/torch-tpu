#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at {
Tensor &triu_out_tpu(const Tensor &self, int64_t diagonal, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(out);
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = triu( self.cpu(), diagonal );
    out = out_cpu.to(out.device());
#else
  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnTriangularizeAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      1,
      diagonal,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::TRIU);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("triu.out", triu_out_tpu); }

// Tensor & triu_indices_out_tpu(int64_t row, int64_t col, int64_t offset, Tensor & out) {
//     CPU_IMPL_WARNING();
//     auto out_cpu = torch::tril_indices(row)
// }
// Tensor triu_indices_tpu(int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {

// }
// TORCH_LIBRARY_IMPL(aten, TPU, m)
// {
//     m.impl("triu_indices.out", triu_indices_out_tpu);
//     m.impl("triu_indices",     triu_indices_tpu);
// }

} // namespace at