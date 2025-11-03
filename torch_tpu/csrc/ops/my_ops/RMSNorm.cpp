#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

#ifdef USING_PPL
#include "RMSNorm.h"

template <typename scalar_t>
    static void rmsnorm_forward_impl(
    uint64_t output_addr,
    uint64_t input_addr,
    uint64_t scale_addr,
    uint64_t bias_addr,
    uint32_t outer_size,
    uint32_t inner_size,
    float eps)
    {
    auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
            uint32_t tile_size) -> int {
        if constexpr (std::is_same_v<scalar_t, float>) {
            return rmsnorm_fp32(
                stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, scale_addr, bias_addr,
                eps, (scale_addr != 0), (bias_addr != 0),
                static_cast<uint32_t>(outer_size),
                static_cast<uint32_t>(inner_size),
                tile_size);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            return rmsnorm_fp16(
                stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, scale_addr, bias_addr,
                eps, (scale_addr != 0), (bias_addr != 0),
                static_cast<uint32_t>(outer_size),
                static_cast<uint32_t>(inner_size),
                tile_size);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            return rmsnorm_bf16(
                stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, scale_addr, bias_addr,
                eps, (scale_addr != 0), (bias_addr != 0),
                static_cast<uint32_t>(outer_size),
                static_cast<uint32_t>(inner_size),
                tile_size);
        }
        return -1;
    };

    auto stream = c10_tpu::getCurrentTPUStream();
    tpuKernelModule_t ppl_module = getPplModule();
    uint32_t tile_size = inner_size;

    while (tile_size >= 1) {
        int ret = kernel(stream, ppl_module, tile_size);
        if (ret == 0) {
            return;
        } else {
            tile_size = tile_size / 2;
            continue;
        }
    }

    TORCH_CHECK(false, "Tile size reduction failed after attempts");
}
#endif

namespace at
{
    Tensor rmsnorm_forward(
        Tensor &input,
        const c10::optional<Tensor> &scale,
        const c10::optional<Tensor> &bias,
        Tensor &output,
        int64_t axis,
        double_t eps)
    {
        TIMING_START;
        CHECK_TENSOR_IN_DEVICE(input);
        CHECK_TENSOR_IN_DEVICE(output);
        if(scale.has_value()){
            CHECK_TENSOR_IN_DEVICE(scale.value());
        }
        if(bias.has_value()){
            CHECK_TENSOR_IN_DEVICE(bias.value());
        }
#ifdef USING_PPL
        if (usePPLKernels())
        {
          uint32_t outer_size = 1;
          uint32_t inner_size = 1;
          for (const auto i : c10::irange(axis)) {
              outer_size *= input.size(i);
          }
          for (const auto i : c10::irange(axis, input.dim())) {
              inner_size *= input.size(i);
          }
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::kHalf, at::kBFloat16, input.scalar_type(), "rmsnorm_forward", [&] {
                  rmsnorm_forward_impl<scalar_t>(
                      reinterpret_cast<uint64_t>(output.data_ptr()),
                      reinterpret_cast<uint64_t>(input.data_ptr()),
                      scale ? reinterpret_cast<uint64_t>(scale->data_ptr()) : 0,
                      bias ? reinterpret_cast<uint64_t>(bias->data_ptr()) : 0,
                      outer_size, inner_size,
                      static_cast<float>(eps));
              });
        } else
#endif
        {
          auto stream = c10_tpu::getCurrentTPUStream();
          tpudnnStatus_t status = tpudnnRmsNormForwardAsync(
              stream,
              tpu::TPUGenerateTpudnnTensor(stream, input),
              scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, scale.value()) : tpudnnUndefinedTensor(),
              bias.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, bias.value()) : tpudnnUndefinedTensor(),
              tpu::TPUGenerateTpudnnTensor(stream, output),
              axis,
              eps);
          TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        }

        TIMING_END;
        return output;
    }

    Tensor rmsnorm_backward(
        Tensor &grad_output,
        Tensor &input,
        const c10::optional<Tensor> &scale,
        const c10::optional<Tensor> &bias,
        Tensor &rms,
        const c10::optional<Tensor> &grad_input,
        const c10::optional<Tensor> &grad_scale,
        const c10::optional<Tensor> &grad_bias,
        int64_t axis,
        double_t eps)
    {
        TIMING_START;
        CHECK_TENSOR_IN_DEVICE(grad_output);
        CHECK_TENSOR_IN_DEVICE(input);
        if (scale.has_value())
        {
            CHECK_TENSOR_IN_DEVICE(scale.value());
        }
        if (bias.has_value())
        {
            CHECK_TENSOR_IN_DEVICE(bias.value());
        }
        CHECK_TENSOR_IN_DEVICE(rms);
        if (grad_input.has_value())
        {
            CHECK_TENSOR_IN_DEVICE(grad_input.value());
        }
        if (grad_scale.has_value())
        {
            CHECK_TENSOR_IN_DEVICE(grad_scale.value());
        }
        if (grad_bias.has_value())
        {
            CHECK_TENSOR_IN_DEVICE(grad_bias.value());
        }

          auto stream = c10_tpu::getCurrentTPUStream();
        tpudnnStatus_t status = tpudnnRmsNormBackwardAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, grad_output),
            tpu::TPUGenerateTpudnnTensor(stream, input),
            scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, scale.value()) : tpudnnUndefinedTensor(),
            tpu::TPUGenerateTpudnnTensor(stream, rms),
            grad_input.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, grad_input.value()) : tpudnnUndefinedTensor(),
            grad_scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, grad_scale.value()) : tpudnnUndefinedTensor(),
            grad_bias.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, grad_bias.value()) : tpudnnUndefinedTensor(),
            axis,
            eps);
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);

        TIMING_END;
        return rms;
    }

}
