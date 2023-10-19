#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <limits.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>


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
  auto grad_input_cpu = threshold_backward ( grad_output.cpu(), input.cpu(), threshold );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnReLUBackward(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(grad_output),
      tpu::TPUGenerateSgdnnTensor(input),
      tpu::TPUGenerateSgdnnTensor(grad_input));
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::RELU_BACKWARD, timer.ElapsedUS());
#endif
#endif
  return grad_input;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("threshold_backward.grad_input", threshold_backward_grad_input_tpu);
}

Tensor &gelu_out_tpu(const Tensor &self, c10::string_view approximate,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = gelu ( self.cpu(), approximate );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status =
      sgdnnGELU(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::GELU, timer.ElapsedUS());
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("gelu.out", gelu_out_tpu); }

Tensor &gelu_backward_grad_input_tpu(const Tensor &grad_output,
                                     const Tensor &self,
                                     c10::string_view approximate,
                                     Tensor &grad_input) {
  CHECK_TENSOR_IN_DEVICE(grad_output);
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(grad_input);
#if 0
  auto grad_input_cpu = gelu_backward ( grad_output.cpu(), self.cpu(), approximate );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnGELUBackward(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(grad_output),
      tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(grad_input));
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::GELU_BACKWARD, timer.ElapsedUS());
#endif
#endif
  return grad_input;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("gelu_backward.grad_input", gelu_backward_grad_input_tpu);
}

Tensor &silu_out_tpu(const Tensor &self, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  LOG( WARNING ) << "silu use cpu impl";
  auto out_cpu = silu(self.to(torch::kFloat).cpu());
  out = out_cpu.to(out.device()).to(out.dtype());
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status =
      sgdnnActive(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                  tpu::TPUGenerateSgdnnTensor(out), ACTIVE_SILU);
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::SILU, timer.ElapsedUS());
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("silu.out", silu_out_tpu); }

Tensor &leakyrelu__tpu(Tensor &self, Scalar negative_slope) {
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "LeakyReLU " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
  auto self_cpu = self.cpu();
  self_cpu = leakyRelu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnLeakyReLU(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(self), negative_slope.to<double>());
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::LEAKY_RELU, timer.ElapsedUS());
#endif
  return self;
#endif
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("leakyRelu_", leakyrelu__tpu); }

Tensor &leakyrelu_tpu(const Tensor &self, const Scalar &negative_slope,
                      Tensor &out) {
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "LEAKYRELU " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
  auto self_cpu = self.cpu();
  self_cpu = leakyRelu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnLeakyReLU(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(out), negative_slope.to<double>());
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::LEAKY_RELU, timer.ElapsedUS());
#endif
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

TIMING_START
    bm_status_t status = sgdnnHardtanh(
                            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                            min_value.toFloat(), max_value.toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
TIMING_END(tpu::REPEAT)

  }

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

Tensor & sigmoid_backward_out_tpu(const Tensor & grad_output, const Tensor & output, Tensor & grad_input){
  CHECK_TENSOR_IN_DEVICE(grad_output);
  CHECK_TENSOR_IN_DEVICE(output);
  CHECK_TENSOR_IN_DEVICE(grad_input);
#if 1
  LOG( WARNING ) << "Sigmoid backward use cpu impl";
  auto grad_input_cpu = sigmoid_backward ( grad_output.cpu(), output.cpu() );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
#endif
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sigmoid_backward.grad_input", sigmoid_backward_out_tpu);
}

} // namespace at
