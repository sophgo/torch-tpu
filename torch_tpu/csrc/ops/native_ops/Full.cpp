#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{
    Tensor full_tpu(IntArrayRef size, const Scalar &fill_value,
                    c10::optional<ScalarType> dtype,
                    c10::optional<Layout> layout,
                    c10::optional<Device> device,
                    c10::optional<bool> pin_memory)
    {
        auto self = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
        CHECK_TENSOR_IN_DEVICE(self);
#if 0
        auto self_cpu = full(size, fill_value, dtype, layout);
        tpu::TPUCopyHostToDevice(self.data_ptr(), self_cpu.contiguous().data_ptr(), self_cpu.nbytes());
#else
        if (size.size() == 0)
        {
            CPU_IMPL_WARNING();
            TIMING_START;
            auto self_cpu = full(size, fill_value, dtype, layout, c10::nullopt, pin_memory);
            tpu::TPUCopyHostToDevice(self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes());
            TIMING_END(tpu::CPU_LAYER);
        }
        else if (IS_TPU_TENSOR(self))
        {
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
            TIMING_START;

            auto stream = c10_tpu::getCurrentTPUStream();
            auto status = tpudnnFillAsync(
                stream,
                &value_,
                tpu::TPUGenerateTpudnnTensor(stream, self));
            TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
                        TIMING_END(tpu::FULL);
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        SHOW_TENSOR_OP(self);
        return self;
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("full", full_tpu);
    }

} // namespace at
