#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include <iostream>
#include "common/config.h"

namespace at
{

    Tensor & real_out_tpu (const Tensor &self, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto self_cpu = neg ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            CPU_IMPL_WARNING();
            TIMING_START;
            auto self_cpu = real(self.cpu());
            tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(), out.nbytes());
            TIMING_END(tpu::CPU_LAYER);
        }
        else if (IS_TPU_TENSOR(self))
        {
            TIMING_START;
            auto stream = c10_tpu::getCurrentTPUStream();
            auto status = tpudnnRealAsync(
                stream,
                tpu::TPUGenerateTpudnnTensorforComplex64(stream, self),
                tpu::TPUGenerateTpudnnTensorforComplex64(stream, out));
            TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
                        TIMING_END(tpu::REAL);
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        SHOW_TENSOR_OP(self, out);
        return out;
    }

    Tensor real_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options().dtype(at::kFloat));
        return real_out_tpu(self, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("real", real_tpu);
    }

} // namespace at

