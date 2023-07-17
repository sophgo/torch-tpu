#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

//#define TPU_OP_TIMING

namespace at
{
    Tensor & bernouli_float_tpu(
            Tensor& self, double p, c10::optional<at::Generator> generator=c10::nullopt) {
#if 1
    auto out_ = bernoulli(self.cpu(), p, generator);
    self = out_.to(self.device());
#endif
    return self;
    }


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "bernoulli_.float", bernouli_float_tpu );
}
} // namespace at