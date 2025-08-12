#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at {
Tensor where_self_tpu(const Tensor &condition, const Tensor &self,
                      const Tensor &other) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(condition);
  auto condition_ = condition.contiguous();
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  TORCH_CHECK(condition_.dim() > 0);
  std::vector<int64_t> sizes_vec(condition_.dim());
  for (auto i = 0; i < condition_.dim(); ++i) {
    sizes_vec[i] = condition_.size(i);
  }
  if (self.dim() > 0) {
    TORCH_CHECK(condition_.dim() == self.dim());
    for (auto i = 0; i < self.dim(); ++i) {
      sizes_vec[i] = std::max(sizes_vec[i], self.size(i));
    }
  }
  if (other.dim() > 0) {
    TORCH_CHECK(condition_.dim() == other.dim());
    for (auto i = 0; i < other.dim(); ++i) {
      sizes_vec[i] = std::max(sizes_vec[i], other.size(i));
    }
  }
  TensorOptions options =
      TensorOptions(condition_.device()).dtype(self.dtype());
  IntArrayRef sizes(sizes_vec.data(), sizes_vec.size());
  auto out = torch::empty(sizes, options);
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnWhereAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, condition_),
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, other),
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(condition, self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("where.self", where_self_tpu); }

} // namespace at