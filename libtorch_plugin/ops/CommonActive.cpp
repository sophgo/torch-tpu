#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {
#define HACK_CPU_IMP 0
#define TPU_OP_TIMING_START

#ifdef TPU_OP_TIMING
#define TIMING_START auto timer = tpu::Timer().Start();
#define TIMING_END(TIMING_NAME)                                                \
  tpu::OpTimer::Instance().AddTime(TIMING_NAME, timer.ElapsedUS());
#else
#define TIMING_START
#define TIMING_END(OP)
#endif

#define IMP_ACTIVE(OP)                                                         \
  Tensor OP##_tpu(const Tensor &self) {                                        \
    auto out = empty(self.sizes(), self.options());                            \
    return OP##_out_tpu(self, out);                                            \
  }

#define IMP_ACTIVE_OUT(OP, ACTIVE_TYPE, TIMING_NAME)                           \
  Tensor &OP##_out_tpu(const Tensor &self, Tensor &out) {                      \
    if (self.dim() > 0) {                                                      \
      CHECK_TENSOR_IN_DEVICE(self);                                            \
    }                                                                          \
    CHECK_TENSOR_IN_DEVICE(out);                                               \
    if (HACK_CPU_IMP) {                                                        \
      auto self_cpu = OP(self.cpu());                                          \
      tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(),  \
                               self.nbytes());                                 \
    } else {                                                                   \
      if (self.dim() == 0) {                                                   \
        auto self_cpu = OP(self.cpu());                                        \
        tpu::TPUCopyHostToDevice(self.data_ptr(),                              \
                                 self.contiguous().data_ptr(), self.nbytes()); \
      } else if (IS_TPU_TENSOR(self)) {                                        \
        TIMING_START                                                           \
        bm_status_t status = sgdnnActive(                                      \
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),      \
            tpu::TPUGenerateSgdnnTensor(out), ACTIVE_TYPE);                    \
        TORCH_CHECK(status == BM_SUCCESS);                                     \
        TIMING_END(TIMING_NAME)                                                \
      } else {                                                                 \
        TORCH_CHECK(false, "At least one input is required in TPU device");    \
      }                                                                        \
    }                                                                          \
    return out;                                                                \
  }

IMP_ACTIVE_OUT(tan, ACTIVE_TAN, tpu::TAN_FORWARD)
IMP_ACTIVE(tan)
IMP_ACTIVE_OUT(cos, ACTIVE_COS, tpu::COS_FORWARD)
IMP_ACTIVE(cos)
IMP_ACTIVE_OUT(sin, ACTIVE_SIN, tpu::SIN_FORWARD)
IMP_ACTIVE(sin)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("tan.out", tan_out_tpu);
  m.impl("cos.out", cos_out_tpu);
  m.impl("sin.out", sin_out_tpu);
  m.impl("tan", tan_tpu);
  m.impl("cos", cos_tpu);
  m.impl("sin", sin_tpu);
}

IMP_ACTIVE_OUT(asinh, ACTIVE_ARCSINH, tpu::ASINH_FORWARD)
IMP_ACTIVE(asinh)
IMP_ACTIVE_OUT(acosh, ACTIVE_ARCCOSH, tpu::ACOSH_FORWARD)
IMP_ACTIVE(acosh)
IMP_ACTIVE_OUT(atanh, ACTIVE_ARCTANH, tpu::ATANH_FORWARD)
IMP_ACTIVE(atanh)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("atanh.out", atanh_out_tpu);
  m.impl("atanh", atanh_tpu);
  m.impl("acosh.out", acosh_out_tpu);
  m.impl("acosh", acosh_tpu);
  m.impl("asinh.out", asinh_out_tpu);
  m.impl("asinh", asinh_tpu);
}

IMP_ACTIVE_OUT(abs, ACTIVE_ABSVAL, tpu::ABS_FORWARD)
IMP_ACTIVE(abs)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("abs.out", abs_out_tpu);
  m.impl("abs", abs_tpu);
}

} // namespace at
