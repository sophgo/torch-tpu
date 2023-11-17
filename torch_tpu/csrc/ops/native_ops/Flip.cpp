#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/cpu/moments_utils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>
// #include "tpu_kernel.h"

namespace at {

Tensor &flip_out_tpu(const Tensor &self, const c10::ArrayRef<int64_t> dims,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() <= 0 || self.dim() > 4 || dims.size() <= 0 || self.dim() < dims.size()) {
    auto self_cpu = flip(self.cpu(), dims);
    tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  }
#if 0
  auto self_cpu = flip(self.cpu(), dims);
  tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                           out.nbytes());
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto temp_result = self;
  for (uint i = 0; i < dims.size(); i++) {
    bm_status_t status = sgdnnFlip(tpu::TPUGetDeviceHandle(),
                                   tpu::TPUGenerateSgdnnTensor(temp_result),
                                   dims[i], tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    temp_result = out;
  }
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::FLIP, timer.ElapsedUS());
#endif
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
