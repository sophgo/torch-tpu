
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <torch/library.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"

namespace at_tpu {
namespace native {

at::Tensor wrapper_Tensor_tpu_format_cast(const at::Tensor & self, const at::Tensor & dst) {
  return at_tpu::TPUNativeFunctions::tpu_format_cast(self, dst);
}

at::Tensor wrapper__tpu_format_cast(const at::Tensor & self, int64_t tpu_format) {
  return at_tpu::TPUNativeFunctions::tpu_format_cast(self, tpu_format);
}


TORCH_LIBRARY(tpu, m) {
  m.def("tpu_format_cast.Tensor(Tensor self, Tensor dst) -> Tensor");
  m.def("tpu_format_cast(Tensor self, int tpu_format) -> Tensor");
}

TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m) {
  m.impl("tpu_format_cast.Tensor", TORCH_FN(at_tpu::native::wrapper_Tensor_tpu_format_cast));
  m.impl("tpu_format_cast", TORCH_FN(at_tpu::native::wrapper__tpu_format_cast));
}

TORCH_LIBRARY_IMPL(tpu, AutogradPrivateUse1, m) {
  m.impl("tpu_format_cast.Tensor", TORCH_FN(at_tpu::native::wrapper_Tensor_tpu_format_cast));
  m.impl("tpu_format_cast", TORCH_FN(at_tpu::native::wrapper__tpu_format_cast));
}


} // namespace native
} // namespace at_tpu
