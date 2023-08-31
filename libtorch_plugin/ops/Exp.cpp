#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor &exp_out_tpu(const Tensor &self, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

  auto self_cpu = exp ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
  if (self.dim() == 0) {
    auto self_cpu = exp(self.cpu());
    tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(),
                             self.nbytes());
  } else if (IS_TPU_TENSOR(self)) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnExp(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                 tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::EXP_FORWARD, timer.ElapsedUS());
#endif
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
}

Tensor exp_tpu(const Tensor &self) {
  auto out = empty(self.sizes(), self.options());
  return exp_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("exp.out", exp_out_tpu);
  m.impl("exp", exp_tpu);
}

Tensor &expm1_out_tpu(const Tensor &self, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

  auto self_cpu = expm1 ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
  if (self.dim() == 0) {
    auto self_cpu = expm1(self.cpu());
    tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(),
                             self.nbytes());
  } else if (IS_TPU_TENSOR(self)) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnExpm1(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                   tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::EXPM1, timer.ElapsedUS());
#endif
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
}

Tensor expm1_tpu(const Tensor &self) {
  auto out = empty(self.sizes(), self.options());
  return expm1_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("expm1.out", expm1_out_tpu);
  m.impl("expm1", expm1_tpu);
}

} // namespace at