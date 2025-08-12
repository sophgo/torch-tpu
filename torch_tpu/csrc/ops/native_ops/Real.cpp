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
    TIMING_START;
    if (self.dim() > 0)
    {
        CHECK_TENSOR_IN_DEVICE(self);
    }
    CHECK_TENSOR_IN_DEVICE(out);
#if 0
    auto self_cpu = neg ( self.cpu());
    tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
    if (IS_TPU_TENSOR(self))
    {
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnRealAsync(
            stream,
            tpu::TPUGenerateTpudnnTensorforComplex64(stream, self),
            tpu::TPUGenerateTpudnnTensorforComplex64(stream, out));
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
    else
    {
        TORCH_CHECK(false, "At least one input is required in TPU device");
    }
#endif
    TIMING_END;
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

