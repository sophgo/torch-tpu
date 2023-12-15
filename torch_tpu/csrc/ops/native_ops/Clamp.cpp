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
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
  auto self_ = self.contiguous();
#if 0
    auto out_cpu = clamp ( self.to(torch::kFloat32).cpu(), min, max );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    if (self_.dim() == 0)
    {
        CPU_IMPL_WARNING();
        TIMING_START;
        auto out_cpu = clamp ( self_.to(torch::kFloat32).cpu(), min, max );
        tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
        TIMING_END(tpu::CPU_LAYER);
    }
    else if (IS_TPU_TENSOR(self_)){
        TIMING_START;
        bm_status_t status = sgdnnClamp(
            tpu::TPUGetDeviceHandle(),
            tpu::TPUGenerateSgdnnTensor(self_),
            min.has_value() ? min.value().to<float>() : -std::numeric_limits<float>::infinity(),
            max.has_value() ? max.value().to<float>() : std::numeric_limits<float>::infinity(),
            tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK(status == BM_SUCCESS);
        TIMING_END(tpu::CLAMP);
    }
    else
    {
        TORCH_CHECK(false, "Input is required in TPU device");
    }
#endif
    SHOW_TENSOR_OP(self, out);
    return out;
}

Tensor clamp_tpu( const at::Tensor & self, const c10::optional<at::Scalar> & min,
                    const c10::optional<at::Scalar> & max) {
  auto out = empty(self.sizes(), self.options());
  return clamp_out_tpu ( self, min, max, out );
}

Tensor & clamp_min_out_tpu(const Tensor & self, const Scalar & min, Tensor & out)
{
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = clamp_min( self.to(torch::kFloat).cpu(), min);
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    out = self.clamp(c10::optional<at::Scalar>( min ), c10::nullopt );
#endif
    return out;
}


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "clamp.out",  clamp_out_tpu);
 m.impl ( "clamp",  clamp_tpu);
 m.impl ( "clamp_min.out", clamp_min_out_tpu);
}

} // namespace at