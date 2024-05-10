#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>


#include "TPUTorchUtils.h"
#include "common/config.h"
#include "sgdnn_api.h"

namespace at {

Tensor &repeat_out_tpu(const Tensor &self, const IntArrayRef repeats,
                       Tensor &out) {
  Tensor contiguous_self = self.is_contiguous() ? self : self.contiguous();
  std::vector<int> repeat_times;
  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    Tensor out_cpu = out.cpu();
    repeat_out(out_cpu, self.cpu(), repeats);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    for (int i = 0; i < (int)repeats.size(); ++i) {
      repeat_times.push_back((int)repeats[i]);
    }

    TIMING_START;

    auto status = sgdnnRepeat(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(contiguous_self),
        repeat_times.data(), repeats.size(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
        TIMING_END(tpu::REPEAT)
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
//  - func: repeat(Tensor self, SymInt[] repeats) -> Tensor
Tensor repeat_tpu(const Tensor &self, const IntArrayRef repeats) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
    TORCH_CHECK((int64_t)repeats.size() >= self.dim());
  }

  IntArrayRef size(repeats);
  std::vector<int64_t> size_vec(size.begin(), size.end());
  if (repeats.size() == (size_t)self.dim()) {
    for (int i = 0; i < self.dim(); ++i) {
      size_vec[i] = repeats[i] * self.size(i);
    }
  } else {
    int dist = repeats.size() - self.dim();
    for (int i = 0; i < (int)repeats.size(); ++i) {
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