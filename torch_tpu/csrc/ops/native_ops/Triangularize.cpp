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

} // namespace at