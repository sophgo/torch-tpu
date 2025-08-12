#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ScatterGatherChecks.h>

#include "TPUTorchUtils.h"

#include <c10/util/Logging.h>
#include "common/config.h"

namespace at
{
Tensor &gather_out_tpu(const Tensor &self, int64_t axis, const Tensor &other, bool sparse_grad, Tensor &out )
{
    TIMING_START;
    if (self.dim() > 0)
    {
        CHECK_TENSOR_IN_DEVICE(self);
    }
    CHECK_TENSOR_IN_DEVICE(out);
#if 0
    auto self_cpu = gather ( self.cpu(), axis, other.cpu(), sparse_grad );
    tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
    tpu::TPUCopyHostToDevice ( other.data_ptr(),other.contiguous().data_ptr(), other.nbytes() );
#else
    if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR ( other ))
    {
        //need to consider broadcast later
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnGatherAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, self),
            tpu::TPUGenerateTpudnnTensor(stream, other),
            tpu::TPUGenerateTpudnnTensor(stream, out),
            axis);
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

Tensor gather_tpu(const Tensor &self, int64_t dim, const Tensor &other, bool sparse_grad)
{
    int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
    native::gather_shape_check(self, wrapped_dim, other);
    TORCH_CHECK(other.dtype() == torch::kInt32, "gather's index must be int32 dtype.");

    TensorOptions options = TensorOptions(self.device()).dtype(self.dtype());
    auto out = empty(other.sizes(), options);
    return gather_out_tpu(self, wrapped_dim, other, sparse_grad, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("gather.out", gather_out_tpu);
    m.impl("gather", gather_tpu);
}

} // namespace at
