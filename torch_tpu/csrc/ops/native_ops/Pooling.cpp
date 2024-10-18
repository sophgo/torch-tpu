#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at {
std::tuple<Tensor, Tensor>
max_pool2d_with_indices_tpu(const Tensor &self, IntArrayRef kernel_size,
                            IntArrayRef stride, IntArrayRef padding,
                            IntArrayRef dilation, bool ceil_mode) {
  CHECK_TENSOR_IN_DEVICE(self);
  std::tuple<Tensor, Tensor> outputs;
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  auto outputs_cpu =
      max_pool2d_with_indices(self.to(torch::kFloat32).cpu(), kernel_size,
                              stride, padding, dilation, ceil_mode);
  outputs = std::tuple<Tensor, Tensor>(
      TENSOR_TO_TPU(std::get<0>(outputs_cpu)).to(self.dtype()),
      TENSOR_TO_TPU(std::get<1>(outputs_cpu)));
  TIMING_END(tpu::CPU_LAYER);
#else
  TORCH_CHECK(ceil_mode == false);
  TORCH_CHECK(dilation[0] == 1 && dilation[1] == 1, "DILATION must be one");
  float alpha = 1.f;
  float beta = 0.f;
  PoolingDescriptor_t pooling_desc = {.kh = (int)kernel_size[0],
                                             .kw = (int)kernel_size[1],
                                             .pad_h = (int)padding[0],
                                             .pad_w = (int)padding[1],
                                             .stride_h = (int)stride[0],
                                             .stride_w = (int)stride[1],
                                             .mode = Pooling_MAX};
  int output_h =
      at::native::pooling_output_shape(self.size(2), kernel_size[0], padding[0],
                                       stride[0], dilation[0], ceil_mode);
  int output_w =
      at::native::pooling_output_shape(self.size(3), kernel_size[1], padding[1],
                                       stride[1], dilation[1], ceil_mode);
  auto output =
      empty({self.size(0), self.size(1), output_h, output_w}, self.options());
  TIMING_START;

  auto status = sgdnnPoolingForward(
      tpu::TPUGetDeviceResource(), pooling_desc, &alpha,
      tpu::TPUGenerateSgdnnTensor(self), ADDR_IN_DEVICE(self), &beta,
      tpu::TPUGenerateSgdnnTensor(output), ADDR_IN_DEVICE(output));
  TORCH_CHECK(status == SG_SUCCESS);
  TIMING_END(tpu::MAX_POOLING);
  outputs = std::tuple<Tensor, Tensor>(output, Tensor());
  SHOW_TENSOR_OP(self, output);
#endif
  return outputs;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("max_pool2d_with_indices", max_pool2d_with_indices_tpu);
}

Tensor max_pool2d_with_indices_backward_tpu(
    const Tensor &grad_output, const Tensor &self, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool ceil_mode, const Tensor &indices) {
  auto grad_output_ =
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  CHECK_TENSOR_IN_DEVICE(grad_output_);
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(indices);
  Tensor grad_input;
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  auto grad_input_cpu = max_pool2d_with_indices_backward(
      grad_output_.to(torch::kFloat32).cpu(), self.to(torch::kFloat32).cpu(),
      kernel_size, stride, padding, dilation, ceil_mode, indices.cpu());
  grad_input = TENSOR_TO_TPU(grad_input_cpu).to(self.dtype());
  TIMING_END(tpu::CPU_LAYER);
#else
  TORCH_CHECK(ceil_mode == false);
  TORCH_CHECK(dilation[0] == 1 && dilation[1] == 1, "DILATION must be one");
  float alpha = 1.f;
  float beta = 0.f;
  PoolingDescriptor_t pooling_desc = {.kh = (int)kernel_size[0],
                                      .kw = (int)kernel_size[1],
                                      .pad_h = (int)padding[0],
                                      .pad_w = (int)padding[1],
                                      .stride_h = (int)stride[0],
                                      .stride_w = (int)stride[1],
                                      .mode = Pooling_MAX};
  int output_h =
      at::native::pooling_output_shape(self.size(2), kernel_size[0], padding[0],
                                       stride[0], dilation[0], ceil_mode);
  int output_w =
      at::native::pooling_output_shape(self.size(3), kernel_size[1], padding[1],
                                       stride[1], dilation[1], ceil_mode);
  auto output =
      empty({self.size(0), self.size(1), output_h, output_w}, self.options());
  grad_input = empty(self.sizes(), self.options());

  TIMING_START;

  auto status = sgdnn_pooling_forward(
      tpu::TPUGetDeviceResource(), pooling_desc, &alpha,
      tpu::TPUGenerateSgdnnTensor(self), ADDR_IN_DEVICE(self), &beta,
      tpu::TPUGenerateSgdnnTensor(output), ADDR_IN_DEVICE(output));
  TORCH_CHECK(status == SG_SUCCESS);
  status = sgdnn_pooling_backward(
      tpu::TPUGetDeviceResource(), pooling_desc, &alpha,
      tpu::TPUGenerateSgdnnTensor(output), ADDR_IN_DEVICE(output),
      tpu::TPUGenerateSgdnnTensor(grad_output), ADDR_IN_DEVICE(grad_output),
      tpu::TPUGenerateSgdnnTensor(self), ADDR_IN_DEVICE(self), &beta,
      tpu::TPUGenerateSgdnnTensor(grad_input), ADDR_IN_DEVICE(grad_input));
  TORCH_CHECK(status == SG_SUCCESS);
  TIMING_END(tpu::MAX_POOLING);
#endif
  SHOW_TENSOR_OP(grad_output_, self, grad_input);
  return grad_input;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("max_pool2d_with_indices_backward",
         max_pool2d_with_indices_backward_tpu);
}

Tensor &avg_pool2d_out_tpu(const Tensor &self, IntArrayRef kernel_size,
                           IntArrayRef stride, IntArrayRef padding,
                           bool ceil_mode, bool count_include_pad,
                           c10::optional<int64_t> divisor_override,
                           Tensor &output) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(output);
#if 0
  auto output_cpu = avg_pool2d ( self.to ( torch::kFloat32 ).cpu(), kernel_size, stride, padding, ceil_mode, count_include_pad );
  output = output_cpu.to(output.device()).to(output.dtype());
#else
  TORCH_CHECK(ceil_mode == false);
  int output_h = at::native::pooling_output_shape(
      self.size(2), kernel_size[0], padding[0], stride[0],
      (long int)1 /*dilation.h*/, ceil_mode);
  int output_w = at::native::pooling_output_shape(
      self.size(3), kernel_size[1], padding[1], stride[1],
      (long int)1 /*dilation.w*/, ceil_mode);
  TPUDNN_PoolingDescriptor_t pooling_desc = {.kh = (int)kernel_size[0],
                                             .kw = (int)kernel_size[1],
                                             .pad_h = (int)padding[0],
                                             .pad_w = (int)padding[1],
                                             .stride_h = (int)stride[0],
                                             .stride_w = (int)stride[1],
                                             .output_h = output_h,
                                             .output_w = output_w,
                                             .mode = TPUDNN_POOLING_AVG};
  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnPoolingForwardAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, output), pooling_desc);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::AVG_POOLING);
#endif
  SHOW_TENSOR_OP(self, output);
  return output;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("avg_pool2d.out", avg_pool2d_out_tpu);
}

Tensor adaptive_avg_pool2d_out_tpu(const Tensor &self,
                                   IntArrayRef output_size) {
  CHECK_TENSOR_IN_DEVICE(self);
  auto output =
      empty({self.size(0), self.size(1), output_size[0], output_size[1]},
            self.options());
#if 0
  auto output_cpu = adaptive_avg_pool2d ( self.to ( torch::kFloat32 ).cpu(), output_size );
  output = output_cpu.to(output.device()).to(output.dtype());
#else
  // padding need to be checked
  std::vector<int32_t> strides(2, 0);
  std::vector<int32_t> kernel_shape(2, 0);
  std::vector<int32_t> pads(2, 0);
  std::vector<int32_t> paddings(2, 0);

  for (int i = 0; i < 2; i++) {
    strides[i] = std::floor(self.size(i + 2) / output_size[i]);
    kernel_shape[i] = self.size(i + 2) - (output_size[i] - 1) * strides[i];
    strides[i] = output_size[i] == 1 ? 1 : strides[i];
  }

  TPUDNN_PoolingDescriptor_t pooling_desc = {.kh = kernel_shape[0],
                                             .kw = kernel_shape[1],
                                             .pad_h = paddings[0],
                                             .pad_w = paddings[1],
                                             .stride_h = strides[0],
                                             .stride_w = strides[1],
                                             .output_h = (int)output_size[0],
                                             .output_w = (int)output_size[1],
                                             .mode = TPUDNN_POOLING_AVG};

  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnPoolingForwardAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, output), pooling_desc);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::AVG_POOLING);
#endif
  SHOW_TENSOR_OP(self, output);
  return output;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("_adaptive_avg_pool2d", adaptive_avg_pool2d_out_tpu);
}
} // namespace at