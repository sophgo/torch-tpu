#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"

namespace at 
{
Tensor & uniform_tpu(Tensor & self, double from, double to, c10::optional<at::Generator> generator)
{
#if 1
  CPU_IMPL_WANING();
  auto out_cpu = uniform(self.cpu(), from, to, generator);
  self = out_cpu.to(self.device());
#else

#endif
  SHOW_TENSOR_OP(self);
  return self;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "uniform_", uniform_tpu );
}
} // namespace at
