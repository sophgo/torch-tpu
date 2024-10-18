#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <limits.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at {

Tensor &threshold_backward_grad_input_tpu(const Tensor &grad_output,
                                          const Tensor &input,
                                          const Scalar &threshold,
                                          Tensor &grad_input) {
  CHECK_TENSOR_IN_DEVICE(grad_output);
  CHECK_TENSOR_IN_DEVICE(input);
  CHECK_TENSOR_IN_DEVICE(grad_input);
#if 0
  CPU_IMPL_WARNING();
  TIMING_START;
  auto grad_input_cpu = threshold_backward ( grad_output.cpu(), input.cpu(), threshold );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
  TIMING_END(tpu::CPU_LAYER);
#else
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnReLUBackwardAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, grad_output),
    tpu::TPUGenerateTpudnnTensor(stream, input),
    tpu::TPUGenerateTpudnnTensor(stream, grad_input));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::RELU_BACKWARD);
#endif
  SHOW_TENSOR_OP(grad_output, input, grad_input);
  return grad_input;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("threshold_backward.grad_input", threshold_backward_grad_input_tpu);
}

Tensor &gelu_out_tpu(const Tensor &self, c10::string_view approximate,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(out);
#if 0
  auto out_cpu = gelu ( self.cpu(), approximate );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto self_ = self.contiguous();
  out = out.contiguous();
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnGELUAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::GELU);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("gelu.out", gelu_out_tpu); }

Tensor &gelu_backward_grad_input_tpu(const Tensor &grad_output,
                                     const Tensor &self,
                                     c10::string_view approximate,
                                     Tensor &grad_input) {
  CHECK_TENSOR_IN_DEVICE(grad_output);
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  CHECK_TENSOR_IN_DEVICE(grad_input);
#if 0
  auto grad_input_cpu = gelu_backward ( grad_output.cpu(), self.cpu(), approximate );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
  auto self_ = self.contiguous();
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnGELUBackwardAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, grad_output),
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, grad_input));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::GELU_BACKWARD);
#endif
  SHOW_TENSOR_OP(grad_output, self, grad_input);
  return grad_input;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("gelu_backward.grad_input", gelu_backward_grad_input_tpu);
}

Tensor &silu_out_tpu(const Tensor &self, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  CPU_IMPL_WARNING();
  auto out_cpu = silu(self.to(torch::kFloat).cpu());
  out = out_cpu.to(out.device()).to(out.dtype());
#else
  auto self_ = self.contiguous();
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnActiveAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, out),
    TPUDNN_ACTIVE_SILU);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::SILU);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("silu.out", silu_out_tpu); }

Tensor &leakyrelu__tpu(Tensor &self, Scalar negative_slope) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
#if 0
  auto self_cpu = self.cpu();
  self_cpu = leakyRelu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
  auto self_ = self.contiguous();
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnLeakyReLUAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    negative_slope.to<double>());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::LEAKY_RELU);
  SHOW_TENSOR_OP(self);
  return self;
#endif
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("leakyRelu_", leakyrelu__tpu); }

Tensor &leakyrelu_tpu(const Tensor &self, const Scalar &negative_slope,
                      Tensor &out) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
#if 0
  auto self_cpu = self.cpu();
  self_cpu = leakyRelu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
  auto self_ = self.contiguous();
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnLeakyReLUAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, out),
    negative_slope.to<double>());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::LEAKY_RELU);
  SHOW_TENSOR_OP(self, out);
  return out;
#endif
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("leaky_relu.out", leakyrelu_tpu); }

Tensor & hardtanh_out_tpu(const Tensor &self, const Scalar &min_value,
                          const Scalar &max_value, Tensor &out) {
  if(self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0) {
    Tensor out_cpu = torch::nn::functional::detail::hardtanh(
                            self.cpu(), min_value.toDouble(), max_value.toDouble(), false);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else {
    auto self_ = self.contiguous();
    TIMING_START;
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnHardtanhAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self_),
      min_value.toFloat(),
      max_value.toFloat(),
      tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::HARDTANH);
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
Tensor hardtanh_tpu(const Tensor &self, const Scalar &min_value,
                      const Scalar &max_value) {
  auto out = empty_like(self);
  return hardtanh_out_tpu(self, min_value, max_value, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("hardtanh", hardtanh_tpu);
  m.impl("hardtanh.out", hardtanh_out_tpu);
}

//TODO TPU implement
Tensor & sigmoid_backward_out_tpu(const Tensor & grad_output, const Tensor & output, Tensor & grad_input){
  CHECK_TENSOR_IN_DEVICE(grad_output);
  CHECK_TENSOR_IN_DEVICE(output);
  CHECK_TENSOR_IN_DEVICE(grad_input);
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  auto grad_input_cpu = sigmoid_backward ( grad_output.cpu(), output.cpu() );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
  TIMING_END(tpu::CPU_LAYER);
#else
#endif
  SHOW_TENSOR_OP(grad_output, output, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sigmoid_backward.grad_input", sigmoid_backward_out_tpu);
}

} // namespace at
