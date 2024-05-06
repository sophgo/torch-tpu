/**
 * Supported operation list:
 * + ACTIVE_TANH = 0,
 * + ACTIVE_SIGMOID = 1, // re-implemented
 * + ACTIVE_RELU = 2, // re-implemented
 * + ACTIVE_EXP = 3,
 * - ACTIVE_ELU = 4,
 * + ACTIVE_SQRT = 5, // re-implemented
 * - ACTIVE_SQUARE = 6,
 * + ACTIVE_RSQRT = 7,
 * + ACTIVE_ABSVAL = 8,
 * - ACTIVE_LN = 9,
 * + ACTIVE_ROUND = 10,
 * + ACTIVE_CEIL = 11,
 * + ACTIVE_FLOOR = 12,
 * + ACTIVE_SIN = 13,
 * + ACTIVE_COS = 14,
 * + ACTIVE_IS_FINITE = 15,
 * - ACTIVE_MISH = 16,
 * - ACTIVE_SWISH = 17,
 * - ACTIVE_HSWISH = 18,
 * - ACTIVE_SILU = 19,
 * + ACTIVE_ARCSIN = 20,
 * + ACTIVE_ARCCOS = 21,
 * + ACTIVE_ARCSINH = 22,
 * + ACTIVE_ARCCOSH = 23,
 * + ACTIVE_ARCTANH = 24,
 * + ACTIVE_SINH = 25,
 * + ACTIVE_COSH = 26,
 * + ACTIVE_TAN = 27,
 * + ACTIVE_SIGN = 28,
 * + ACTIVE_GELU = 29,
 * + ACTIVE_ERF = 30,
 * - ACTIVE_HSIGMOID = 31,
 * - ACTIVE_LOG_SIGMOID = 32,
 * - ACTIVE_SOFT_PLUS = 33,
 * + ACTIVE_SOFT_SIGN = 34,
 * ******** currently only implemented in tpu-train ********
 * + ACTIVE_ERFC = 35,
 * + ACTIVE_ISNAN = 35, **new implementation**
 * + ACTIVE_ISINF = 35, **new implementation**
 * + ACTIVE_EXPM1 = 38,
 * + ACTIVE_RECIPROCAL = 39,  **new implementation**(Reciprocal.cpp)
 * + ACTIVE_EXP2 = 40, **new implementation**
 * + ACTIVE_TRUNC = 41, **new implementation**
 */
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {
#define HACK_CPU_IMP 0

#define IMP_ACTIVE(OP)                                                         \
  Tensor OP##_tpu(const Tensor &self) {                                        \
    auto out = empty(self.sizes(), self.options());                            \
    return OP##_out_tpu(self, out);                                            \
  }

#define IMP_ACTIVE_BOOL(OP)                                                         \
  Tensor OP##_tpu(const Tensor &self) {                                        \
    auto out = empty(self.sizes(), self.options().dtype(at::kBool));                            \
    return OP##_out_tpu(self, out);                                            \
  }

#define IMP_ACTIVE_(OP)                                                        \
  Tensor &OP##__tpu(Tensor &self) { return OP##_out_tpu(self, self); }

#if defined BACKEND_1684X 
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
        TIMING_START;                                                          \
        auto status = sgdnnActive(                                             \
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),      \
            tpu::TPUGenerateSgdnnTensor(out), ACTIVE_TYPE);                    \
        TORCH_CHECK(status == BM_SUCCESS);                                     \
        TIMING_END(TIMING_NAME)                                                \
      } else {                                                                 \
        TORCH_CHECK(false, "At least one input is required in TPU device");    \
      }                                                                        \
    }                                                                          \
    SHOW_TENSOR_OP(self, out);                                                 \
    return out;                                                                \
  }
#elif defined BACKEND_SG2260
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
        TIMING_START;                                                          \
        auto status = sgdnnActive(                                             \
            c10_tpu::getCurrentTPUStream(), tpu::TPUGenerateSgdnnTensor(self), \
            tpu::TPUGenerateSgdnnTensor(out), ACTIVE_TYPE, true);              \
        TORCH_CHECK(status == tpuRtSuccess);                                     \
        TIMING_END(TIMING_NAME)                                                \
      } else {                                                                 \
        TORCH_CHECK(false, "At least one input is required in TPU device");    \
      }                                                                        \
    }                                                                          \
    SHOW_TENSOR_OP(self, out);                                                 \
    return out;                                                                \
  }
#endif

IMP_ACTIVE_OUT(abs, ACTIVE_ABSVAL, tpu::ABS_FORWARD)
IMP_ACTIVE(abs)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("abs.out", abs_out_tpu);
  m.impl("abs", abs_tpu);
}

IMP_ACTIVE_OUT(exp, ACTIVE_EXP, tpu::EXP_FORWARD)
IMP_ACTIVE(exp)
IMP_ACTIVE_OUT(exp2, ACTIVE_EXP2, tpu::EXP2)
IMP_ACTIVE(exp2)
IMP_ACTIVE_OUT(expm1, ACTIVE_EXPM1, tpu::EXPM1)
IMP_ACTIVE(expm1)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("exp.out", exp_out_tpu);
  m.impl("exp", exp_tpu);
  m.impl("exp2.out", exp2_out_tpu);
  m.impl("exp2", exp2_tpu);
  m.impl("expm1.out", expm1_out_tpu);
  m.impl("expm1", expm1_tpu);
}

IMP_ACTIVE_OUT(erf, ACTIVE_ERF, tpu::ERF)
IMP_ACTIVE(erf)
IMP_ACTIVE_OUT(erfc, ACTIVE_ERFC, tpu::ERFC)
IMP_ACTIVE(erfc)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("erf.out", erf_out_tpu);
  m.impl("erf", erf_tpu);
  m.impl("erfc.out", erfc_out_tpu);
  m.impl("erfc", erfc_tpu);
}

IMP_ACTIVE_OUT(ceil, ACTIVE_CEIL, tpu::CEIL)
IMP_ACTIVE(ceil)
IMP_ACTIVE_OUT(floor, ACTIVE_FLOOR, tpu::FLOOR)
IMP_ACTIVE(floor)
IMP_ACTIVE_OUT(round, ACTIVE_ROUND, tpu::ROUND)
IMP_ACTIVE(round)
IMP_ACTIVE_OUT(trunc, ACTIVE_TRUNC, tpu::TRUNC)
IMP_ACTIVE(trunc)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ceil", ceil_tpu);
  m.impl("ceil.out", ceil_out_tpu);
  m.impl("floor", floor_tpu);
  m.impl("floor.out", floor_out_tpu);
  m.impl("round", round_tpu);
  m.impl("round.out", round_out_tpu);
  m.impl("trunc", trunc_tpu);
  m.impl("trunc.out", trunc_out_tpu);
}

IMP_ACTIVE_OUT(sqrt, ACTIVE_SQRT, tpu::SQRT)
IMP_ACTIVE(sqrt)
IMP_ACTIVE_OUT(rsqrt, ACTIVE_RSQRT, tpu::RSQRT)
IMP_ACTIVE(rsqrt)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sqrt.out", sqrt_out_tpu);
  m.impl("sqrt", sqrt_tpu);
  m.impl("rsqrt.out", rsqrt_out_tpu);
  m.impl("rsqrt", rsqrt_tpu);
}

IMP_ACTIVE_OUT(isfinite, ACTIVE_IS_FINITE, tpu::ISFINITE)
IMP_ACTIVE(isfinite)
IMP_ACTIVE_OUT(isnan, ACTIVE_ISNAN, tpu::ISNAN)
IMP_ACTIVE_BOOL(isnan)
IMP_ACTIVE_OUT(isinf, ACTIVE_ISINF, tpu::ISINF)
IMP_ACTIVE_BOOL(isinf)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("isfinite.out", isfinite_out_tpu);
  m.impl("isfinite", isfinite_tpu);
  m.impl("isinf.out", isinf_out_tpu);
  m.impl("isinf", isinf_tpu);
  m.impl("isnan.out", isnan_out_tpu);
  m.impl("isnan", isnan_tpu);
}

IMP_ACTIVE_OUT(sign, ACTIVE_SIGN, tpu::SIGN)
IMP_ACTIVE(sign)
IMP_ACTIVE_OUT(relu, ACTIVE_RELU, tpu::RELU)
IMP_ACTIVE_(relu)
IMP_ACTIVE(relu)
IMP_ACTIVE_OUT(sigmoid, ACTIVE_SIGMOID, tpu::SIGMOID)
IMP_ACTIVE(sigmoid)

// TODO: should take f16 and backward support  into count
// IMP_ACTIVE_OUT(gleu, ACTIVE_GELU, tpu::SIGN)
// IMP_ACTIVE(gleu)
// IMP_ACTIVE_OUT(silu, ACTIVE_SILU, tpu::SILU)
// IMP_ACTIVE(silu)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sign.out", sign_out_tpu);
  m.impl("sign", sign_tpu);
  m.impl("relu.out", relu_out_tpu);
  m.impl("relu_", relu__tpu);
  m.impl("relu", relu_tpu);
  m.impl("sigmoid.out", sigmoid_out_tpu);
  m.impl("sigmoid", sigmoid_tpu);
  // m.impl("silu.out", silu_out_tpu);
  // m.impl("silu", silu_tpu);
  // m.impl("gleu.out", gleu_out_tpu);
  // m.impl("gleu", gleu_tpu);
}

IMP_ACTIVE_OUT(sin, ACTIVE_SIN, tpu::SIN_FORWARD)
IMP_ACTIVE(sin)
IMP_ACTIVE_OUT(cos, ACTIVE_COS, tpu::COS_FORWARD)
IMP_ACTIVE(cos)
IMP_ACTIVE_OUT(tan, ACTIVE_TAN, tpu::TAN_FORWARD)
IMP_ACTIVE(tan)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sin.out", sin_out_tpu);
  m.impl("sin", sin_tpu);
  m.impl("cos.out", cos_out_tpu);
  m.impl("cos", cos_tpu);
  m.impl("tan.out", tan_out_tpu);
  m.impl("tan", tan_tpu);
}

IMP_ACTIVE_OUT(asin, ACTIVE_ARCSIN, tpu::ASIN)
IMP_ACTIVE(asin)
IMP_ACTIVE_OUT(acos, ACTIVE_ARCCOS, tpu::ACOS)
IMP_ACTIVE(acos)
IMP_ACTIVE_OUT(atan, ACTIVE_ARCTANH, tpu::ATAN)
IMP_ACTIVE(atan)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("asin.out", asin_out_tpu);
  m.impl("asin", asin_tpu);
  m.impl("acos.out", acos_out_tpu);
  m.impl("acos", acos_tpu);
  m.impl("atan.out", atan_out_tpu);
  m.impl("atan", atan_tpu);
}

IMP_ACTIVE_OUT(asinh, ACTIVE_ARCSINH, tpu::ASINH_FORWARD)
IMP_ACTIVE(asinh)
IMP_ACTIVE_OUT(acosh, ACTIVE_ARCCOSH, tpu::ACOSH_FORWARD)
IMP_ACTIVE(acosh)
IMP_ACTIVE_OUT(atanh, ACTIVE_ARCTANH, tpu::ATANH_FORWARD)
IMP_ACTIVE(atanh)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("asinh.out", asinh_out_tpu);
  m.impl("asinh", asinh_tpu);
  m.impl("acosh.out", acosh_out_tpu);
  m.impl("acosh", acosh_tpu);
  m.impl("atanh.out", atanh_out_tpu);
  m.impl("atanh", atanh_tpu);
}

IMP_ACTIVE_OUT(sinh, ACTIVE_SINH, tpu::SINH_FORWARD)
IMP_ACTIVE(sinh)
IMP_ACTIVE_OUT(cosh, ACTIVE_COSH, tpu::COSH_FORWARD)
IMP_ACTIVE(cosh)
IMP_ACTIVE_OUT(tanh, ACTIVE_TANH, tpu::TANH_FORWARD)
IMP_ACTIVE(tanh)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sinh.out", sinh_out_tpu);
  m.impl("sinh", sinh_tpu);
  m.impl("cosh.out", cosh_out_tpu);
  m.impl("cosh", cosh_tpu);
  m.impl("tanh.out", tanh_out_tpu);
  m.impl("tanh", tanh_tpu);
}

} // namespace at
