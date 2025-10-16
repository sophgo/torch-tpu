#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"
#ifdef USING_PPL
#include "Fill.h"
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
static void full_async_impl(
	uint64_t output_addr,
  scalar_t value,
  int outer_size
  ){
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
        uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return fill_fp32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return fill_fp16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return fill_bf16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
        return fill_int32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
        return fill_uint8(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    }
    return -1;
  };

	auto stream = c10_tpu::getCurrentTPUStream();
	tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = outer_size;
  while (tile_size >= 1) {
      int ret = kernel(stream, ppl_module, tile_size);
      if (ret == 0) {
          return;
      } else {
          tile_size = tile_size / 2;
          continue;
      }
  }
	TORCH_CHECK(false, "full failed !");
}
#endif
namespace at
{
    Tensor full_tpu(IntArrayRef size, const Scalar &fill_value,
                    c10::optional<ScalarType> dtype,
                    c10::optional<Layout> layout,
                    c10::optional<Device> device,
                    c10::optional<bool> pin_memory)
    {
        TIMING_START;
#if 0
        auto self_cpu = full(size, fill_value, dtype, layout);
        tpu::TPUCopyHostToDevice(self.data_ptr(), self_cpu.contiguous().data_ptr(), self_cpu.nbytes());
#else
        auto self = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
        int64_t value_;
        if (self.dtype() == caffe2::TypeMeta::Make<float>())
        {
            auto value = fill_value.toFloat();
            memcpy(&value_, &value, sizeof(float));
        }
        else if (self.dtype() == caffe2::TypeMeta::Make<at::Half>())
        {
            auto fp16 = fill_value.toHalf();
            memcpy(&value_, &fp16, sizeof(at::Half));
        }
        else if (self.dtype() == caffe2::TypeMeta::Make<at::BFloat16>())
        {
            auto bf16 = fill_value.toBFloat16();
            memcpy(&value_, &bf16, sizeof(at::BFloat16));
        }
            else if (self.dtype() == caffe2::TypeMeta::Make<int>())
        {
            auto value = fill_value.toInt();
            memcpy(&value_, &value, sizeof(int));
        }
        else
        {
            TORCH_CHECK(false);
        }
#ifdef USING_PPL
        if (usePPLKernels()) {
            int outer_size = 1;
            for (const auto i : c10::irange(self.dim())) {
                outer_size *= self.size(i);
            }
            AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "full", [&] {
                    full_async_impl<scalar_t>(
                        reinterpret_cast<uint64_t>(self.data_ptr()),
                        fill_value.to<scalar_t>(),
                        outer_size
                        );
                });
        } else
#endif
        {
            auto stream = c10_tpu::getCurrentTPUStream();
            auto status = tpudnnFillAsync(
                stream,
                &value_,
                tpu::TPUGenerateTpudnnTensor(stream, self));
            TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        }
#endif
        TIMING_END;
        SHOW_TENSOR_OP(self);
        return self;
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("full", full_tpu);
    }

} // namespace at
