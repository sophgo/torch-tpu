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

    Tensor &logical_or_out_tpu(const Tensor &self, const Tensor &other, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
 
  auto self_cpu = logical_or ( self.cpu(),other.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
  tpu::TPUCopyHostToDevice ( other.data_ptr(),other.contiguous().data_ptr(), other.nbytes() );
#else
        if (self.dim() == 0 && other.dim() == 0)
        {
            CPU_IMPL_WARNING();
            TIMING_START;
            auto out_cpu = logical_or(self.cpu(),other.cpu());
            tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
            TIMING_END(tpu::CPU_LAYER);
        }
        else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR ( other ))
        {
        //need to consider broadcast later
            TIMING_START;
            bm_status_t status = sgdnnLogicalOr(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(other),
                tpu::TPUGenerateSgdnnTensor(out));
            TORCH_CHECK(status == BM_SUCCESS);
            TIMING_END(tpu::LOGICAL_OR);
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        SHOW_TENSOR_OP(self, other, out);
        return out;
    }

    Tensor logical_or_tpu(const Tensor &self, const Tensor &other)
    {
        auto out = empty(self.sizes(), self.options());
        return logical_or_out_tpu(self, other, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("logical_or.out", logical_or_out_tpu);
        m.impl("logical_or", logical_or_tpu);
    }

} // namespace at
