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
Tensor &neg_out_tpu(const Tensor &self, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    auto out_cpu = neg(self.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  }

#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status =
      sgdnnNeg(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
               tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::NEG, timer.ElapsedUS());
#endif

  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("neg.out", neg_out_tpu); }
} // namespace at
