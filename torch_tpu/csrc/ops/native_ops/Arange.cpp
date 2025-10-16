#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

#ifdef USING_PPL
#include "Arange.h"
#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, func)  \
AT_DISPATCH_SWITCH(                   \
scalar_type, name,                    \
AT_DISPATCH_CASE(at::kFloat, func)    \
AT_DISPATCH_CASE(at::kHalf, func)     \
AT_DISPATCH_CASE(at::kBFloat16, func) \
AT_DISPATCH_CASE(at::kInt, func)      \
AT_DISPATCH_CASE(at::kShort, func)    \
AT_DISPATCH_CASE(at::kChar, func)     \
AT_DISPATCH_CASE(at::kByte, func))

template <typename scalar_t>
static void arange_async_impl(
    uint64_t output_addr,
    uint32_t start,
    uint32_t end,
    uint32_t step,
    int output_shape
  )
{
    // only support single core current, the same with nodechip_arange.
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
        return arange_int32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, start, end, step, output_shape
            );
    } else if constexpr (std::is_same_v<scalar_t, float>) {
        return arange_fp32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, start, end, step, output_shape
            );
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return arange_fp16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, start, end, step, output_shape
            );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return arange_bf16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, start, end, step, output_shape
            );
    }
    return -1;
  };

    auto stream = c10_tpu::getCurrentTPUStream();
    tpuKernelModule_t ppl_module = getPplModule();
    int ret = kernel(stream, ppl_module);
    if (ret == 0) {
        return;
    }
    TORCH_CHECK(false, "arange failed !");
}
#endif
namespace at
{
Tensor & arange_start_out_tpu( const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out)
{
    TIMING_START;
    CHECK_TENSOR_IN_DEVICE ( out );
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = arange(start,end,step);
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    if ((start.toInt() >= 0 && end.toInt() >= 0)){
#ifdef USING_PPL
        if (usePPLKernels())
        {
            int out_shape = out.size(0);
            AT_DISPATCH_FLOAT_INT_TYPES( out.scalar_type(), "arange", [&] {
                    arange_async_impl<scalar_t>(
                        reinterpret_cast<uint64_t>(out.data_ptr()),
                        start.toInt(),
                        end.toInt(),
                        step.toInt(),
                        out_shape
                        );
                });
        } else
    #endif
        {   auto stream = c10_tpu::getCurrentTPUStream();
            auto status = tpudnnArangeAsync(
                stream,
                start.toInt(),
                end.toInt(),
                step.toInt(),
                tpu::TPUGenerateTpudnnTensor(stream, out)
                );
            TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        }

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
                                                                                        ScalarType::Int : end.type()
                                                            ).layout(layout);
    if ( device.has_value()) {
        TORCH_CHECK( device.value().is_privateuseone() );
        options = options.device( device );
    }
    if ( dtype.has_value() ) {
        TORCH_CHECK( (dtype.value() == torch::kInt32) || (dtype.value() == torch::kFloat32) ||
                     (dtype.value() == torch::kFloat16) || (dtype.value() == torch::kBFloat16),
                    "arange only support int32 & float32 now" );
        options = options.dtype ( dtype );
    }

    TORCH_CHECK( (start.isIntegral(false) && end.isIntegral(false)) ||
                 (start.toDouble() - (int)start.toDouble() == 0.0 && end.toDouble() - (int)end.toDouble() == 0.0),
                 "arange Decimal places are not supported now" )

    int empty_length = (end.toInt()-start.toInt() - 1) / step.toInt() + 1;
    auto out = empty({empty_length}, options);
    out = arange_start_out_tpu(start, end, step, out);
    return out;
}

Tensor arange_start_tpu(const Scalar & start, const Scalar & end, c10::optional<ScalarType> dtype,
                c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) {
    auto out = arange_start_step_tpu(start, end, 1, dtype, layout, device, pin_memory);
    return out;
}

Tensor arange_tpu(const Scalar & end, c10::optional<ScalarType> dtype, c10::optional<Layout> layout,
                c10::optional<Device> device, c10::optional<bool> pin_memory) {
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

