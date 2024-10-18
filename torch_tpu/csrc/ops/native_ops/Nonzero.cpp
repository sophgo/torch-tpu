#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

namespace at {

Tensor nonzero_tpu(const Tensor &self) {
  CHECK_TENSOR_IN_DEVICE(self);
  int size = 1;
  for (int i=0; i<self.dim(); i++){
    size *= self.size(i);
  }
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

  Tensor out = empty({num.item().toInt(), self.dim()}, self.options().dtype(kInt));
  for (int i=0; i<num.item().toInt(); ++i){
    for (int j=0; j<self.dim(); ++j){
      out[i][j] = out_temp[i][j];
    }
  }
  // wait tpu support resize_
  // out.resize_((num.item().toInt(), self.dim()), c10::nullopt);
  SHOW_TENSOR_OP(self);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("nonzero", nonzero_tpu); }

} // namespace at