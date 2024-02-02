#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor upsample_bilinear2d_tpu(const at::Tensor &self,
                               at::ArrayRef<long> output_size,
                               bool align_corners,
                               c10::optional<double> scales_h,
                               c10::optional<double> scales_w) {
  CHECK_TENSOR_IN_DEVICE(self);
  TORCH_CHECK(self.dim() > 0, "input dim should larger than 0.");

#if 0
  auto self_cpu = upsample_bilinear2d(self.cpu(), output_size, align_corners,
                                      scale_factors);
  tpu::TPUCopyHostToDevice(out.data_ptr(), self.contiguous().data_ptr(),
                           self.nbytes());
#else
  std::vector<int64_t> output_shape(4, 0);
  output_shape[0] = self.size(0);
  output_shape[1] = self.size(1);
  auto size_ref = output_size;
  output_shape[2] = size_ref[0];
  output_shape[3] = size_ref[1];

  if (!align_corners && scales_h.has_value() && scales_w.has_value()) {
    output_shape[2] = (int64_t)(scales_h.value() * self.size(2));
    output_shape[3] = (int64_t)(scales_w.value() * self.size(3));
  }

  at::IntArrayRef output_shape_ref(output_shape);
  auto out = empty(output_shape_ref, self.options());

  TIMING_START;
  #if defined BACKEND_1684X
  auto status = sgdnnUpsampling(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(out), align_corners, UPSAMPLING_BILINEAR);

  if (scales_h.has_value() && scales_w.has_value()) {
    out = out.slice(2, c10::nullopt, size_ref[0])
              .slice(3, c10::nullopt, size_ref[1]);
  }
  TORCH_CHECK(status == BM_SUCCESS);
  #elif defined BACKEND_SG2260
  auto status = sgdnnUpsampling(
      c10_tpu::getCurrentTPUStream(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(out), align_corners, UPSAMPLING_BILINEAR);

  if (scales_h.has_value() && scales_w.has_value()) {
    out = out.slice(2, c10::nullopt, size_ref[0])
              .slice(3, c10::nullopt, size_ref[1]);
  }
  TORCH_CHECK(status == tpuRtSuccess);
  #endif
  TIMING_END(tpu::UPSAMPLING_BILINEAR)
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("upsample_bilinear2d", upsample_bilinear2d_tpu);
}

Tensor upsample_nearest2d_tpu(const at::Tensor &self,
                              at::ArrayRef<long> output_size,
                              c10::optional<double> scales_h,
                              c10::optional<double> scales_w) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  TORCH_CHECK(self.dim() > 0, "input dim should larger than 0.");
#if 0
  auto self_cpu = upsample_bilinear2d(self.cpu(), output_size, align_corners,
                                      scale_factors);
  tpu::TPUCopyHostToDevice(out.data_ptr(), self.contiguous().data_ptr(),
                           self.nbytes());
#else
  std::vector<int64_t> output_shape(4, 0);
  output_shape[0] = self.size(0);
  output_shape[1] = self.size(1);
  auto size_ref = output_size;
  output_shape[2] = size_ref[0];
  output_shape[3] = size_ref[1];

  at::IntArrayRef output_shape_ref(output_shape);
  auto out = empty(output_shape_ref, self.options());

  if (scales_h.has_value() && scales_w.has_value()) {
    output_shape[2] = (int64_t)(scales_h.value() * self.size(2));
    output_shape[3] = (int64_t)(scales_w.value() * self.size(3));
  }
  auto self_ = self.is_contiguous() ? self : self.contiguous(); 
  TIMING_START;
  #if defined BACKEND_1684X
  auto status = sgdnnUpsampling(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self_),
      tpu::TPUGenerateSgdnnTensor(out), true /*align_corners*/,
      UPSAMPLING_NEAREST);

  if (scales_h.has_value() && scales_w.has_value()) {
    out = out.slice(2, c10::nullopt, size_ref[0])
              .slice(3, c10::nullopt, size_ref[1]);
  }
  TORCH_CHECK(status == BM_SUCCESS);
  #elif defined BACKEND_SG2260
  #endif
  TIMING_END(tpu::UPSAMPLING_NEAREST)
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("upsample_nearest2d", upsample_nearest2d_tpu);
}

auto input = at::zeros(2);
Tensor &upsample_nearest2d_backward_out_tpu(
    const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<double> scales_h = c10::nullopt,
    c10::optional<double> scales_w = c10::nullopt,
    at::Tensor &grad_input = input) {
  CHECK_TENSOR_IN_DEVICE(grad_input);
  CHECK_TENSOR_IN_DEVICE(grad_output);
#if 0
  CPU_IMPL_WARNING();
  auto input_type = grad_input.dtype();
  auto T_input = grad_input.cpu().to(torch::kFloat32).contiguous();
  auto input_r = upsample_nearest2d_backward_outf(
      grad_output.cpu().to(torch::kFloat32), output_size, input_size,
      scales_h, scales_w, T_input);
  tpu::TPUCopyHostToDevice(grad_input.data_ptr(),
                           grad_input.cpu().to(input_type).contiguous().data_ptr(),
                           grad_input.nbytes());
  grad_input = grad_input.contiguous();
#else
  PoolingDescriptor_t pooling_desc =
  {
    .kh = int(scales_h.value()),
    .kw = int(scales_w.value()),
    .pad_h = 0,
    .pad_w = 0,
    .stride_h = int(scales_h.value()),
    .stride_w = int(scales_w.value()),
    .output_h = static_cast<int>(output_size[0]),
    .output_w = static_cast<int>(output_size[1]),
    .mode = POOLING_AVG
  };
  TIMING_START;
  #if defined BACKEND_1684X
  auto status = sgdnnUpsampleNearest2dBackward (
                      tpu::TPUGetDeviceHandle(),
                      tpu::TPUGenerateSgdnnTensor ( grad_output ),
                      tpu::TPUGenerateSgdnnTensor ( grad_input ),
                      scales_h.value() * scales_w.value(),
                      pooling_desc );
  TORCH_CHECK ( status == BM_SUCCESS );
  #elif defined BACKEND_SG2260
  #endif
  TIMING_END(tpu::UPSAMPLING_NEAREST_BACKWARD)
#endif
  SHOW_TENSOR_OP(grad_output, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("upsample_nearest2d_backward.grad_input",
         upsample_nearest2d_backward_out_tpu);
}

} // namespace at
