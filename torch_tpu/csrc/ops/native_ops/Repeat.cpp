#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>


#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at {

Tensor &repeat_out_tpu(const Tensor &self, const IntArrayRef repeats,
                       Tensor &out) {
  TIMING_START;
  if (self.nbytes() == 0) {  TIMING_END; SHOW_TENSOR_OP(self, out); return out;}
  Tensor contiguous_self = self.is_contiguous() ? self : self.contiguous();
  std::vector<int> repeat_times;
#if 0
    CPU_IMPL_WARNING();
    Tensor out_cpu = out.cpu();
    repeat_out(out_cpu, self.cpu(), repeats);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
#else
  for (int i = 0; i < (int)repeats.size(); ++i) {
      repeat_times.push_back((int)repeats[i]);
    }

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnRepeatAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, contiguous_self),
        repeat_times.data(),
        repeats.size(),
        tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
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