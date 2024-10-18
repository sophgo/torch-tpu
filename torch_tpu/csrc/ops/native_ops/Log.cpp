#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor &logx_out_tpu(const Tensor &self, Tensor &out, tensor_log_type_t log_type) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = log(self.cpu());
  tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                           out.nbytes());
#else
  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = log(self.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (IS_TPU_TENSOR(self)) {
    TIMING_START;
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnLogAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, out),
        log_type);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        TIMING_END(tpu::LOG_FORWARD)
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

Tensor logx_tpu(const Tensor &self, tensor_log_type_t log_type) {
  auto out = empty(self.sizes(), self.options());
  return logx_out_tpu(self, out, log_type);
}

Tensor &log_out_tpu(const Tensor &self, Tensor &out) {
  return logx_out_tpu(self, out, TPUDNN_LOG_E);
}

Tensor &log1p_out_tpu(const Tensor &self, Tensor &out) {
  return logx_out_tpu(self, out, TPUDNN_LOG_1P);
}

Tensor &log2_out_tpu(const Tensor &self, Tensor &out) {
  return logx_out_tpu(self, out, TPUDNN_LOG_2);
}

Tensor &log10_out_tpu(const Tensor &self, Tensor &out) {
  return logx_out_tpu(self, out, TPUDNN_LOG_10);
}

Tensor log_tpu(const Tensor &self) {
  auto out = empty(self.sizes(), self.options());
  return log_out_tpu(self, out);
}
Tensor log1p_tpu(const Tensor &self) {
  auto out = empty(self.sizes(), self.options());
  return log1p_out_tpu(self, out);
}
Tensor log2_tpu(const Tensor &self) {
  auto out = empty(self.sizes(), self.options());
  return log2_out_tpu(self, out);
}
Tensor log10_tpu(const Tensor &self) {
  auto out = empty(self.sizes(), self.options());
  return log10_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("log.out", log_out_tpu);
  m.impl("log", log_tpu);
  m.impl("log1p.out", log1p_out_tpu);
  m.impl("log1p", log1p_tpu);
  m.impl("log2.out", log2_out_tpu);
  m.impl("log2", log2_tpu);
  m.impl("log10.out", log10_out_tpu);
  m.impl("log10", log10_tpu);
}

} // namespace at
