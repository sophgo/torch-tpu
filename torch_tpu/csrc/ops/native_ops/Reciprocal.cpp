#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace at {
Tensor &reciprocal_out_tpu(const at::Tensor &self, at::Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = reciprocal ( self.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#endif

  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = reciprocal(self.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START
    /**
     * common implemented active (CommonActive.cpp) call cpu function when
     * self.dim() == 0, but when the op is reciprocal, 1 / tensor meet the
     * self.dim() == 0 condition, which should be handled separately.
     */
    bm_status_t status = sgdnnActive(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        tpu::TPUGenerateSgdnnTensor(out), ACTIVE_RECIPROCAL);
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::RECIPROCAL)
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("reciprocal.out", reciprocal_out_tpu);
}

} // namespace at
