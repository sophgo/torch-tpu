#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/native/CPUFallback.h>

#include "common/config.h"

namespace at
{

void cpu_fallback_for_dynamo(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, TPU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback_for_dynamo>());
}



} // namespace at
