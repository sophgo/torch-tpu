#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor &logx_out_tpu(const Tensor &self, Tensor &out, tensor_log_type_t log_type) {
  TIMING_START;
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = log(self.cpu());
  tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                           out.nbytes());
#else
  if (IS_TPU_TENSOR(self)) {
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnLogAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, out),
        log_type);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  TIMING_END;
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


/*************** logsigmoid ***********************/
std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward_tpu(const at::Tensor & self) {
  TIMING_START;
  // TODO: impl it
  auto out    = empty(self.sizes(), self.options());
  auto buffer = empty(self.sizes(), self.options());
  auto min    = torch::minimum(self, torch::zeros_like(self));
  buffer      = self.abs().neg().exp();
  out         = min - buffer.log1p();
  TIMING_END;
  return {out, buffer};
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("log_sigmoid_forward", log_sigmoid_forward_tpu);
}

} // namespace at
