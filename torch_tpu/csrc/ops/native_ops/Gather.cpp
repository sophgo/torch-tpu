#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ScatterGatherChecks.h>

#include "TPUTorchUtils.h"

#include <c10/util/Logging.h>
#include "common/config.h"

#ifdef USING_PPL
#include "Gather.h"
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
static void gather_async_impl(
    uint64_t input_addr,
    uint64_t index_addr,
    uint64_t output_addr,
    uint32_t outer_size,
    uint32_t inner_size,
    int gather_num,
    int gathered_num
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return gather_fp32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return gather_fp16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return gather_bf16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
        return gather_int32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
        return gather_int16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
        return gather_int8(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
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
	TORCH_CHECK(false, "gather failed !");
}
#endif
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
#ifdef USING_PPL
    if (usePPLKernels())
    {

    uint32_t inner_size = 1;
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(axis)) {
        outer_size *= self.size(i);
    }
    for (const auto i : c10::irange((axis + 1), self.dim())) {
        inner_size *= self.size(i);
    }
    AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "gather", [&] {
            gather_async_impl<scalar_t>(
                reinterpret_cast<uint64_t>(self.data_ptr()),
                reinterpret_cast<uint64_t>(other.data_ptr()),
                reinterpret_cast<uint64_t>(out.data_ptr()),
                outer_size,
                inner_size,
                self.size(axis),
                other.size(axis)
                );
        });

    } else
#endif
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
