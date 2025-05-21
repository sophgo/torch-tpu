#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at {
// ========== clamp
Tensor & clamp_out_tpu( const Tensor & self, const c10::optional<Scalar> & min,
                        const c10::optional<Scalar> & max, Tensor & out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self ); }
  auto self_ = self.contiguous();
  auto out_  = out;
  if ( !out.is_contiguous() ) { out_ = out.contiguous(); LOG( WARNING ) << "clamp out not contiguous"; }

#if 0
    auto out_cpu = clamp ( self.to(torch::kFloat32).cpu(), min, max );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    TIMING_START;
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnClampAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self_),
        min.has_value() ? min.value().to<float>() : -std::numeric_limits<float>::infinity(),
        max.has_value() ? max.value().to<float>() : std::numeric_limits<float>::infinity(),
        tpu::TPUGenerateTpudnnTensor(stream, out_));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::CLAMP);
    if ( !out.is_contiguous() ) { out.copy_(out_); }
#endif
    SHOW_TENSOR_OP(self, out);
    return out;
}

Tensor& clamp_Tensor_out_tpu ( const Tensor & self, const c10::optional<Tensor> & min,
                               const c10::optional<Tensor> & max, Tensor & out )
{
    CPU_IMPL_WARNING();
    auto min_cpu = min.has_value() ? c10::optional<Tensor>(min.value().cpu()) : c10::nullopt;
    auto max_cpu = max.has_value() ? c10::optional<Tensor>(max.value().cpu()) : c10::nullopt;
    auto out_cpu = clamp ( self.to(torch::kFloat32).cpu(), min_cpu, max_cpu );
    out = out_cpu.to(out.device()).to(out.dtype());
    return out;
}

Tensor & clamp__tpu(Tensor & self, const c10::optional<Scalar> & min,
                    const c10::optional<Scalar> & max) 
{
    return clamp_out_tpu(self, min, max, self);
}

Tensor & clamp__Tensor_tpu(Tensor & self, const c10::optional<Tensor> & min,
                const c10::optional<Tensor> & max)
{
    return clamp_Tensor_out_tpu(self, min, max, self);
}

Tensor clamp_tpu( const Tensor & self, const c10::optional<Scalar> & min,
                    const c10::optional<Scalar> & max) {
  auto out = empty(self.sizes(), self.options());
  return clamp_out_tpu ( self, min, max, out );
}

Tensor clamp_Tensor_tpu(const Tensor & self, const c10::optional<Tensor> & min,
                    const c10::optional<Tensor> & max)
{
    auto out = empty(self.sizes(), self.options());
    return clamp_Tensor_out_tpu(self, min, max, out );
}

// ========== clamp_min
Tensor & clamp_min_out_tpu(const Tensor & self, const Scalar & min, Tensor & out)
{
    return clamp_out_tpu(self, c10::optional<Scalar>( min ), c10::nullopt, out);
}

Tensor & clamp_min_Tensor_out_tpu(const Tensor & self, const Tensor & min, Tensor & out)
{
    return clamp_Tensor_out_tpu(self, c10::optional<Tensor>( min ) , c10::nullopt, out);
}

inline Tensor & clamp_min___tpu(Tensor & self, const Scalar & min)
{
    return clamp__tpu(self, c10::optional<Scalar>( min ), c10::nullopt);
}

inline Tensor & clamp_min__Tensor_tpu(Tensor & self, const Tensor & min)
{
    return clamp__Tensor_tpu(self, c10::optional<Tensor>( min ), c10::nullopt);
}

inline Tensor clamp_min_tpu(const Tensor & self, const Scalar & min)
{
    return clamp_tpu(self, c10::optional<Scalar>( min ), c10::nullopt);
}

inline Tensor clamp_min_Tensor_tpu(const Tensor & self, const Tensor & min)
{
    return clamp_Tensor_tpu( self,  c10::optional<Tensor>( min ), c10::nullopt);
}

// ========== clamp_max
Tensor & clamp_max_out_tpu(const Tensor & self, const Scalar & max, Tensor & out)
{
    return clamp_out_tpu(self, c10::nullopt, c10::optional<Scalar>( max ), out);
}

Tensor & clamp_max_Tensor_out_tpu(const Tensor & self, const Tensor & max, Tensor & out)
{
    return clamp_Tensor_out_tpu(self, c10::nullopt, c10::optional<Tensor>( max ), out);
}

inline Tensor & clamp_max___tpu(Tensor & self, const Scalar & max)
{
    return clamp__tpu(self,  c10::nullopt, c10::optional<Scalar>( max ));
}

inline Tensor & clamp_max__Tensor_tpu(Tensor & self, const Tensor & max)
{
    return clamp__Tensor_tpu(self, c10::nullopt, c10::optional<Tensor>( max ));
}

inline Tensor clamp_max_tpu(const Tensor & self, const Scalar & max)
{
    return clamp_tpu(self, c10::nullopt, c10::optional<Scalar>( max ));
}

inline Tensor clamp_max_Tensor_tpu(const Tensor & self, const Tensor & max)
{
    return clamp_Tensor_tpu( self, c10::nullopt, c10::optional<Tensor>( max ));
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "clamp.out",         clamp_out_tpu);
 m.impl ( "clamp.Tensor_out",  clamp_Tensor_out_tpu);
 m.impl ( "clamp_",            clamp__tpu);
 m.impl ( "clamp_.Tensor",     clamp__Tensor_tpu);
 m.impl ( "clamp",             clamp_tpu);
 m.impl ( "clamp.Tensor",      clamp_Tensor_tpu);

 m.impl ( "clamp_min.out",          clamp_min_out_tpu);
 m.impl ( "clamp_min.Tensor_out",   clamp_min_Tensor_out_tpu);
 m.impl ( "clamp_min_",             clamp_min___tpu);
 m.impl ( "clamp_min_.Tensor",      clamp_min__Tensor_tpu);
 m.impl ( "clamp_min",              clamp_min_tpu);
 m.impl ( "clamp_min.Tensor",       clamp_min_Tensor_tpu);

 m.impl ( "clamp_max.out",          clamp_max_out_tpu);
 m.impl ( "clamp_max.Tensor_out",   clamp_max_Tensor_out_tpu);
 m.impl ( "clamp_max_",             clamp_max___tpu);
 m.impl ( "clamp_max_.Tensor",      clamp_max__Tensor_tpu);
 m.impl ( "clamp_max",              clamp_max_tpu);
 m.impl ( "clamp_max.Tensor",       clamp_max_Tensor_tpu);
}

} // namespace at