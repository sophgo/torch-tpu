#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

namespace at {

Tensor nonzero_tpu(const Tensor &self) {
#if 1
  CPU_IMPL_WARNING();
  auto out_cpu = nonzero(self.cpu());
  Tensor out = out_cpu.to(self.device());
#else
  CHECK_TENSOR_IN_DEVICE(self);
  int size = self.numel();
  Tensor out_temp = empty({size, self.dim()}, self.options().dtype(kInt));
  Tensor num = empty({1}, self.options().dtype(kInt));
  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnNonzeroAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, out_temp),
      tpu::TPUGenerateTpudnnTensor(stream, num));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::NONZERO);

  Tensor out = out_temp.index({Slice(0, num.item().toInt()), "..."});
#endif
  // wait tpu support resize_
  // out.resize_((num.item().toInt(), self.dim()), c10::nullopt);
  SHOW_TENSOR_OP(self);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("nonzero", nonzero_tpu); }

} // namespace at