#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUDeviceManager.h"
#include "TPUTorchUtils.h"
#include "common/config.h"
#include "sgdnn_api.h"

namespace at {

//  - func: repeat(Tensor self, SymInt[] repeats) -> Tensor
Tensor repeat_tpu(const Tensor &self, const IntArrayRef repeats) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
    TORCH_CHECK(repeats.size() >= self.dim());
  }
  // some op lead to non-contiguous tensor, so we need to make it contiguous
  Tensor contiguous_self = self.is_contiguous() ? self : self.contiguous();

  torch::Device device(torch::kPrivateUse1);
  IntArrayRef size(repeats);
  std::vector<int> repeat_times;
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
  Tensor out = torch::empty(size, self.dtype()).to(device);

  if (self.dim() == 0) {
    Tensor out_cpu;
    repeat_out(out_cpu, self.cpu(), repeats);
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                               out.nbytes());
  }

  for (int i = 0; i < repeats.size(); ++i) {
    repeat_times.push_back((int)repeats[i]);
  }

  TIMING_START
  bm_status_t status = sgdnnRepeat(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(contiguous_self),
      repeat_times.data(), repeats.size(), tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REPEAT)

  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("repeat", repeat_tpu); }

} // namespace at