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
 * + ACTIVE_ISNEGINF = 43, **new implementation**
 * + ACTIVE_ISPOSINF = 44, **new implementation**
 */
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

#ifdef USING_PPL
#include "sigmoid.h"

template <typename scalar_t>
static void active_impl(
    uint64_t output_addr,
    uint64_t input_addr,
    uint32_t inner_size,
    tensor_active_type_t active_type
){
  if (!(active_type == TPUDNN_ACTIVE_SIGMOID)) {
    TORCH_CHECK(false, "Unsupported active_type: ", active_type);
  }
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
    int tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return activation_sigmoid_impl_float32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, inner_size,
        tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return activation_sigmoid_impl_float16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, inner_size,
        tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return activation_sigmoid_impl_bf16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, inner_size,
        tile_size);
    }
    return -1;
  };

  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();

  constexpr uint32_t LOCAL_MEM_BYTES = 256 * 1024; // 256KB
  constexpr uint32_t ELEMENT_SIZE = sizeof(scalar_t);
  constexpr uint32_t MAX_TILE_SIZE = LOCAL_MEM_BYTES / ELEMENT_SIZE;
  uint32_t tile_size = std::min(inner_size, MAX_TILE_SIZE);
  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size);
    if (ret == 0) {
      return;
    } else {
      tile_size = tile_size / 2;
      continue;
    }
  }
  TORCH_CHECK(false, "Tile size reduction failed after attempts");
}
#endif

namespace at {

#define IMP_ACTIVE(OP)                                                         \
  Tensor OP##_tpu(const Tensor &self) {                                        \
    auto out = empty(self.sizes(), self.options());                            \
    return OP##_out_tpu(self, out);                                            \
  }

#define IMP_ACTIVE_BOOL(OP)                                                    \
  Tensor OP##_tpu(const Tensor &self) {                                        \
    auto out = empty(self.sizes(), self.options().dtype(at::kBool));           \
    return OP##_out_tpu(self, out);                                            \
  }

#define IMP_ACTIVE_(OP)                                                        \
  Tensor &OP##__tpu(Tensor &self) { return OP##_out_tpu(self, self); }

using ActiveOpType = at::Tensor (&)(const at::Tensor&);

template <tensor_active_type_t ActiveType, ActiveOpType op>
Tensor &active_template(const Tensor &self, Tensor &out) {
    if (self.dim() > 0) {
        CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
    }
    CHECK_TENSOR_IN_DEVICE(out);
    if (!IS_TPU_TENSOR(self)) {
        auto self_cpu = op(self.cpu());
          tpu::TPUCopyHostToDevice(out.data_ptr(),
                                   self_cpu.contiguous().data_ptr(),
                                   out.nbytes());
    } else if (IS_TPU_TENSOR(self)) {
        auto self_ = self.contiguous();

#ifdef USING_PPL
        if (usePPLKernels() && ActiveType == TPUDNN_ACTIVE_SIGMOID)
        {
            int length = 1;
            for (const auto i : c10::irange(self_.dim())) {
              length *= self_.size(i);
            }
            AT_DISPATCH_FLOATING_TYPES_AND2(
              at::kHalf, at::kBFloat16, self_.scalar_type(), "active", [&] {
                active_impl<scalar_t>(
                  reinterpret_cast<uint64_t>(out.data_ptr()),
                  reinterpret_cast<uint64_t>(self_.data_ptr()),
                  length,
                 ActiveType
                  );
                });
        } else
#endif
        {
            auto stream = c10_tpu::getCurrentTPUStream();
            auto status = tpudnnActiveAsync(
                  stream,
                  tpu::TPUGenerateTpudnnTensor(stream, self_),
                  tpu::TPUGenerateTpudnnTensor(stream, out),
                  ActiveType);
            TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        }
    } else {
      TORCH_CHECK(false, "At least one input is required in TPU device");
    }

    return out;
}

#define IMP_ACTIVE_OUT(OP, ACTIVE_TYPE, TIMING_NAME)                    \
  Tensor &OP##_out_tpu(const Tensor &self, Tensor &out) {               \
      TIMING_START;                                                     \
      auto &ret = active_template<ACTIVE_TYPE, OP>(self, out);          \
      TIMING_END;                                                       \
      SHOW_TENSOR_OP(self, out);                                        \
      return ret;                                                       \
  }

IMP_ACTIVE_OUT(abs, TPUDNN_ACTIVE_ABSVAL, tpu::ACTIVE)
IMP_ACTIVE(abs)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("abs.out", abs_out_tpu);
  m.impl("abs", abs_tpu);
}

IMP_ACTIVE_OUT(exp, TPUDNN_ACTIVE_EXP, tpu::ACTIVE)
IMP_ACTIVE(exp)
IMP_ACTIVE_OUT(exp2, TPUDNN_ACTIVE_EXP2, tpu::ACTIVE)
IMP_ACTIVE(exp2)
IMP_ACTIVE_OUT(expm1, TPUDNN_ACTIVE_EXPM1, tpu::ACTIVE)
IMP_ACTIVE(expm1)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("exp.out", exp_out_tpu);
  m.impl("exp", exp_tpu);
  m.impl("exp2.out", exp2_out_tpu);
  m.impl("exp2", exp2_tpu);
  m.impl("expm1.out", expm1_out_tpu);
  m.impl("expm1", expm1_tpu);
}

IMP_ACTIVE_OUT(erf, TPUDNN_ACTIVE_ERF, tpu::ACTIVE)
IMP_ACTIVE(erf)
IMP_ACTIVE_OUT(erfc, TPUDNN_ACTIVE_ERFC, tpu::ACTIVE)
IMP_ACTIVE(erfc)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("erf.out", erf_out_tpu);
  m.impl("erf", erf_tpu);
  m.impl("erfc.out", erfc_out_tpu);
  m.impl("erfc", erfc_tpu);
}

IMP_ACTIVE_OUT(ceil, TPUDNN_ACTIVE_CEIL, tpu::ACTIVE)
IMP_ACTIVE(ceil)
IMP_ACTIVE_OUT(floor, TPUDNN_ACTIVE_FLOOR, tpu::ACTIVE)
IMP_ACTIVE(floor)
IMP_ACTIVE_OUT(round, TPUDNN_ACTIVE_ROUND, tpu::ACTIVE)
IMP_ACTIVE(round)
IMP_ACTIVE_OUT(trunc, TPUDNN_ACTIVE_TRUNC, tpu::ACTIVE)
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

IMP_ACTIVE_OUT(sqrt, TPUDNN_ACTIVE_SQRT, tpu::ACTIVE)
IMP_ACTIVE(sqrt)
IMP_ACTIVE_OUT(rsqrt, TPUDNN_ACTIVE_RSQRT, tpu::ACTIVE)
IMP_ACTIVE(rsqrt)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sqrt.out", sqrt_out_tpu);
  m.impl("sqrt", sqrt_tpu);
  m.impl("rsqrt.out", rsqrt_out_tpu);
  m.impl("rsqrt", rsqrt_tpu);
}

IMP_ACTIVE_OUT(isfinite, TPUDNN_ACTIVE_IS_FINITE, tpu::ACTIVE)
IMP_ACTIVE_BOOL(isfinite)
IMP_ACTIVE_OUT(isnan, TPUDNN_ACTIVE_ISNAN, tpu::ACTIVE)
IMP_ACTIVE_BOOL(isnan)
IMP_ACTIVE_OUT(isinf, TPUDNN_ACTIVE_ISINF, tpu::ACTIVE)
IMP_ACTIVE_BOOL(isinf)
IMP_ACTIVE_OUT(isneginf, TPUDNN_ACTIVE_ISNEGINF, tpu::ACTIVE)
IMP_ACTIVE_BOOL(isneginf)
IMP_ACTIVE_OUT(isposinf, TPUDNN_ACTIVE_ISPOSINF, tpu::ACTIVE)
IMP_ACTIVE_BOOL(isposinf)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("isfinite.out", isfinite_out_tpu);
  m.impl("isfinite", isfinite_tpu);
  m.impl("isinf.out", isinf_out_tpu);
  m.impl("isinf", isinf_tpu);
  m.impl("isnan.out", isnan_out_tpu);
  m.impl("isnan", isnan_tpu);
  m.impl("isneginf.out", isneginf_out_tpu);
  m.impl("isneginf", isneginf_tpu);
  m.impl("isposinf.out", isposinf_out_tpu);
  m.impl("isposinf", isposinf_tpu);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("isfinite", isfinite_tpu);
}

IMP_ACTIVE_OUT(sign, TPUDNN_ACTIVE_SIGN, tpu::ACTIVE)
IMP_ACTIVE(sign)
IMP_ACTIVE_OUT(relu, TPUDNN_ACTIVE_RELU, tpu::ACTIVE)
IMP_ACTIVE_(relu)
IMP_ACTIVE(relu)
IMP_ACTIVE_OUT(sigmoid, TPUDNN_ACTIVE_SIGMOID, tpu::ACTIVE)
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

IMP_ACTIVE_OUT(sin, TPUDNN_ACTIVE_SIN, tpu::ACTIVE)
IMP_ACTIVE(sin)
IMP_ACTIVE_OUT(cos, TPUDNN_ACTIVE_COS, tpu::ACTIVE)
IMP_ACTIVE(cos)
IMP_ACTIVE_OUT(tan, TPUDNN_ACTIVE_TAN, tpu::ACTIVE)
IMP_ACTIVE(tan)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sin.out", sin_out_tpu);
  m.impl("sin", sin_tpu);
  m.impl("cos.out", cos_out_tpu);
  m.impl("cos", cos_tpu);
  m.impl("tan.out", tan_out_tpu);
  m.impl("tan", tan_tpu);
}

IMP_ACTIVE_OUT(asin, TPUDNN_ACTIVE_ARCSIN, tpu::ACTIVE)
IMP_ACTIVE(asin)
IMP_ACTIVE_OUT(acos, TPUDNN_ACTIVE_ARCCOS, tpu::ACTIVE)
IMP_ACTIVE(acos)
IMP_ACTIVE_OUT(atan, TPUDNN_ACTIVE_ARCTAN, tpu::ACTIVE)
IMP_ACTIVE(atan)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("asin.out", asin_out_tpu);
  m.impl("asin", asin_tpu);
  m.impl("acos.out", acos_out_tpu);
  m.impl("acos", acos_tpu);
  m.impl("atan.out", atan_out_tpu);
  m.impl("atan", atan_tpu);
}

IMP_ACTIVE_OUT(asinh, TPUDNN_ACTIVE_ARCSINH, tpu::ACTIVE)
IMP_ACTIVE(asinh)
IMP_ACTIVE_OUT(acosh, TPUDNN_ACTIVE_ARCCOSH, tpu::ACTIVE)
IMP_ACTIVE(acosh)
IMP_ACTIVE_OUT(atanh, TPUDNN_ACTIVE_ARCTANH, tpu::ACTIVE)
IMP_ACTIVE(atanh)

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("asinh.out", asinh_out_tpu);
  m.impl("asinh", asinh_tpu);
  m.impl("acosh.out", acosh_out_tpu);
  m.impl("acosh", acosh_tpu);
  m.impl("atanh.out", atanh_out_tpu);
  m.impl("atanh", atanh_tpu);
}

IMP_ACTIVE_OUT(sinh, TPUDNN_ACTIVE_SINH, tpu::ACTIVE)
IMP_ACTIVE(sinh)
IMP_ACTIVE_OUT(cosh, TPUDNN_ACTIVE_COSH, tpu::ACTIVE)
IMP_ACTIVE(cosh)
IMP_ACTIVE_OUT(tanh, TPUDNN_ACTIVE_TANH, tpu::ACTIVE)
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
