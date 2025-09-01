#include <ATen/autocast_mode.h>
#include <iostream>
#include <exception>

namespace {
using namespace at::autocast;

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
  KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(matmul, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(convolution_overrideable, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(convolution_backward_overrideable, lower_precision_fp)
  // f32
}

}//namespace