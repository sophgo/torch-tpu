#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUDeviceManager.h"
#include "TPUTorchUtils.h"
#include "common/config.h"
#include "sgdnn_api.h"

namespace at {

Tensor &repeat_out_tpu(const Tensor &self, const IntArrayRef repeats,
                       Tensor &out) {
  Tensor contiguous_self = self.is_contiguous() ? self : self.contiguous();
  std::vector<int> repeat_times;
  if (self.dim() == 0) {
    Tensor out_cpu = out.cpu();
    repeat_out(out_cpu, self.cpu(), repeats);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {
    for (int i = 0; i < repeats.size(); ++i) {
      repeat_times.push_back((int)repeats[i]);
    }

    TIMING_START
    bm_status_t status = sgdnnRepeat(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(contiguous_self),
        repeat_times.data(), repeats.size(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::REPEAT)
  }

  return out;
}
//  - func: repeat(Tensor self, SymInt[] repeats) -> Tensor
Tensor repeat_tpu(const Tensor &self, const IntArrayRef repeats) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
    TORCH_CHECK(repeats.size() >= self.dim());
  }

  IntArrayRef size(repeats);
  std::vector<int64_t> size_vec(size.begin(), size.end());
  if (repeats.size() == self.dim()) {
    for (int i = 0; i < self.dim(); ++i) {
      size_vec[i] = repeats[i] * self.size(i);
    }
  } else {
    int dist = repeats.size() - self.dim();
    for (int i = 0; i < repeats.size(); ++i) {
      if (i < dist) {
        continue;
      } else {
        size_vec[i] *= self.size(i - dist);
      }
    }
  }
  size = torch::IntArrayRef(size_vec);
  auto out = empty(size, self.options());
  repeat_out_tpu(self, repeats, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("repeat.out", repeat_out_tpu);
  m.impl("repeat", repeat_tpu);
}

} // namespace at