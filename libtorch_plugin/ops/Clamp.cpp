#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"

namespace at {
Tensor & clamp_out_tpu( const at::Tensor & self, const c10::optional<at::Scalar> & min,
                        const c10::optional<at::Scalar> & max, at::Tensor & out) {
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 1
    auto out_cpu = clamp ( self.to(torch::kFloat32).cpu(), min, max );
    out = out_cpu.to(out.device()).to(out.dtype());
#endif

    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "clamp.out",  clamp_out_tpu);
}

} // namespace at