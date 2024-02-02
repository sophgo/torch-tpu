#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

namespace at {

Tensor &flip_out_tpu(const Tensor &self, const c10::ArrayRef<int64_t> dims,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() <= 0 || self.dim() > 4 || dims.size() <= 0 || self.dim() < (int)dims.size()) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto self_cpu = flip(self.cpu(), dims);
    tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    return out;
  }
#if 0
  auto self_cpu = flip(self.cpu(), dims);
  tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                           out.nbytes());
#else
  TIMING_START;
  #if defined BACKEND_1684X
  auto temp_result = self;
  for (uint i = 0; i < dims.size(); i++) {
    auto status = sgdnnFlip(tpu::TPUGetDeviceHandle(),
                                   tpu::TPUGenerateSgdnnTensor(temp_result),
                                   dims[i], tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    temp_result = out;
  }
  #elif defined BACKEND_SG2260
  auto temp_result = self;
  for (uint i = 0; i < dims.size(); i++) {
    auto status = sgdnnFlip(c10_tpu::getCurrentTPUStream(),
                                   tpu::TPUGenerateSgdnnTensor(temp_result),
                                   dims[i], tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == tpuRtSuccess);
    temp_result = out;
  }
  #endif
  TIMING_END(tpu::FLIP);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

Tensor flip_tpu(const Tensor &self, c10::ArrayRef<int64_t> dims) {
  auto out = at::empty(self.sizes(), self.options());
  return flip_out_tpu(self, dims, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("flip", flip_tpu); }
} // namespace at
