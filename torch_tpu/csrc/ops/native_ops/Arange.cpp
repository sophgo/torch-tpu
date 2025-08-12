#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
Tensor & arange_start_out_tpu( const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out)
{
    TIMING_START;
    // LOG( WARNING ) << __func__ ;
    CHECK_TENSOR_IN_DEVICE ( out );
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = arange(start,end,step);
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    if ((start.toInt() >= 0 && end.toInt() >= 0)){
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnArangeAsync(
            stream,
            start.toInt(),
            end.toInt(),
            step.toInt(),
            tpu::TPUGenerateTpudnnTensor(stream, out)
            );
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }else{
        CPU_IMPL_WARNING();
        auto out_cpu = arange(start,end,step);
        out = out_cpu.to(out.device()).to(out.dtype());
    }
#endif
    TIMING_END;
    SHOW_TENSOR_OP(out);
    return out;
}

Tensor arange_start_step_tpu(const Scalar & start, const Scalar & end, const Scalar & step, c10::optional<ScalarType> dtype,
                c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
    TensorOptions options = TensorOptions(DeviceType::PrivateUse1).dtype(
                                                            end.type() == ScalarType::Double ? 
                                                                ScalarType::Float : end.type() == ScalarType::Long ?
                                                                                        ScalarType::Int : ScalarType::Int
                                                            ).layout(layout);
    if ( device.has_value()) { 
        TORCH_CHECK( device.value().is_privateuseone() );
        options = options.device( device );
    }
    if ( dtype.has_value() ) { 
        TORCH_CHECK( (dtype.value() == torch::kInt32) || (dtype.value() == torch::kFloat32),
                    "arange only support int32 & float32 now" );
        options = options.dtype ( dtype );
    }

    TORCH_CHECK( (start.isIntegral() && end.isIntegral()) ||
                 (start.toDouble() - (int)start.toDouble() == 0.0 && end.toDouble() - (int)end.toDouble() == 0.0),
                 "arange Decimal places are not supported now" )

    int empty_length = (end.toInt()-start.toInt() - 1) / step.toInt() + 1;
    auto out = empty({empty_length}, options);
    out = arange_start_out_tpu(start, end, step, out);
    return out;
}

Tensor arange_start_tpu(const Scalar & start, const Scalar & end, c10::optional<ScalarType> dtype,
                c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
    // LOG( WARNING ) << __func__ ;
    auto out = arange_start_step_tpu(start, end, 1, dtype, layout, device, pin_memory);
    return out;
}

Tensor arange_tpu(const Scalar & end, c10::optional<ScalarType> dtype, c10::optional<Layout> layout,
                c10::optional<Device> device, c10::optional<bool> pin_memory) {
    // LOG( WARNING ) << __func__ ;
    auto out = arange_start_step_tpu(0, end, 1, dtype, layout, device, pin_memory);
    return out;
}


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "arange.start_out",  arange_start_out_tpu);
 m.impl ( "arange.start_step", arange_start_step_tpu);
 m.impl ( "arange.start",      arange_start_tpu);
 m.impl ( "arange",            arange_tpu);
}
}

