#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

using c10::OperatorHandle;
using c10::Stack;
using c10::DispatchKey;
using c10::DispatchKeySet;
using c10::Dispatcher;
using c10::KernelFunction;

namespace {
using namespace torch::autograd;
// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}
}