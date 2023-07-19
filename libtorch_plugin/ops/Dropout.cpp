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
    Tensor dropout_tpu(const at::Tensor & input, double p, bool train) {
#if 1
    auto input_cpu =input.cpu();
    auto out_cpu = dropout(input_cpu, p, train);
    auto output_tpu = out_cpu.to(input.device());
#endif
    return output_tpu;
    }

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "dropout", dropout_tpu );
}
TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
{
  m.impl ( "dropout", dropout_tpu );
}


    Tensor & dropout__tpu(at::Tensor & input, double p, bool train) {
#if 1
    auto input_cpu =input.cpu();
    auto out_cpu = dropout_(input_cpu, p, train);
    input = out_cpu.to(input.device());
#endif
    return input;
    }

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "dropout_", dropout__tpu );
}
TORCH_LIBRARY_IMPL ( aten, AutogradPrivateUse1, m )
{
  m.impl ( "dropout_", dropout__tpu );
}

} // namespace at