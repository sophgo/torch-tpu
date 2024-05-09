#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{

    Tensor &logical_not_out_tpu(const Tensor &self, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0

  auto self_cpu = logical_not ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            TIMING_START;
            auto self_cpu = exp(self.cpu());
            tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(), self.nbytes());
            TIMING_END(tpu::CPU_LAYER);
        }
        else if (IS_TPU_TENSOR(self))
        {
            TIMING_START;

            auto status = sgdnnLogicalNot(
                tpu::TPUGetDeviceResource(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(out));
            TORCH_CHECK(status == SG_SUCCESS);
                        TIMING_END(tpu::LOGICAL_NOT);
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        SHOW_TENSOR_OP(self, out);
        return out;
    }

    Tensor logical_not_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return logical_not_out_tpu(self, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("logical_not.out", logical_not_out_tpu);
        m.impl("logical_not", logical_not_tpu);
    }

} // namespace at
