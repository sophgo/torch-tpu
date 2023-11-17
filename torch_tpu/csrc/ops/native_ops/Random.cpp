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
Tensor & random_from_to_tpu(Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator){
#if 1
  CPU_IMPL_WANING();
  auto out_cpu = random(self.cpu(), from, to, generator);
  self = out_cpu.to(self.device());
#else

#endif
  SHOW_TENSOR_OP(self);
  return self;
}

Tensor& random_to_tpu(Tensor& self, int64_t to, c10::optional<Generator> generator)
{
  return random_from_to_tpu(self, 0, to, generator);
}

Tensor& random__tpu(Tensor& self, c10::optional<Generator> generator)
{
#if 1
  LOG( WARNING ) << "random_ use cpu impl";
  auto out_cpu = random(self, generator);
  self = out_cpu.to(self.device());
#else

#endif
  SHOW_TENSOR_OP(self);
  return self;
}


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl( "random_.from", random_from_to_tpu );
  m.impl( "random_.to", random_to_tpu );
  m.impl( "random_", random__tpu );
}
} // namespace at