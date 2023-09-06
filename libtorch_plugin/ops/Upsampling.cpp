#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
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

  TIMING_START

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

  bm_status_t status = sgdnnUpsampling(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(out), align_corners, UPSAMPLING_BILINEAR);

  if (scales_h.has_value() && scales_w.has_value()) {
    out = out.slice(2, c10::nullopt, size_ref[0])
              .slice(3, c10::nullopt, size_ref[1]);
  }
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::UPSAMPLING_BILINEAR)

#endif
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("upsample_bilinear2d", upsample_bilinear2d_tpu);
}

Tensor upsample_nearest2d_tpu(const at::Tensor &self,
                              at::ArrayRef<long> output_size,
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

  TIMING_START

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

  bm_status_t status = sgdnnUpsampling(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(out), true /*align_corners*/,
      UPSAMPLING_NEAREST);

  if (scales_h.has_value() && scales_w.has_value()) {
    out = out.slice(2, c10::nullopt, size_ref[0])
              .slice(3, c10::nullopt, size_ref[1]);
  }
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::UPSAMPLING_NEAREST)

#endif
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("upsample_nearest2d", upsample_nearest2d_tpu);
}

} // namespace at
