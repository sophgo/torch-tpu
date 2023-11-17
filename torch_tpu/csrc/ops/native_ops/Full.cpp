#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

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
            auto self_cpu = full(size, fill_value, dtype, layout, c10::nullopt, pin_memory);
            tpu::TPUCopyHostToDevice(self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes());
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            int64_t value_;
            if (self.dtype() == caffe2::TypeMeta::Make<float>())
            {
                *(float *)(&value_) = fill_value.toFloat();
            }
            else if (self.dtype() == caffe2::TypeMeta::Make<at::Half>())
            {
                *(at::Half *)(&value_) = fill_value.toHalf();
            }
            else if (self.dtype() == caffe2::TypeMeta::Make<at::BFloat16>())
            {
                *(at::BFloat16 *)(&value_) = fill_value.toBFloat16();
            }
             else if (self.dtype() == caffe2::TypeMeta::Make<int>())
            {
                *(int *)(&value_) = fill_value.toInt();
            }
            else
            {
                TORCH_CHECK(false);
            }

            bm_status_t status = sgdnnFill(
                tpu::TPUGetDeviceHandle(),
                &value_,
                tpu::TPUGenerateSgdnnTensor(self));
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::FULL, timer.ElapsedUS());
#endif
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
