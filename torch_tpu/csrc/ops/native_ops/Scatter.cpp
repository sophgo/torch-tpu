#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>

#include "TPUTorchUtils.h"
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

#define CONSTANT (0)
#define REFLECT (1)
#define SYMMETRIC (2)
#define REPLICATE (3)
#define CIRCULAR (4)

#ifdef USING_PPL
#include "Scatter.h"
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
static void slice_scatter_impl(
  uint64_t param_addr,
  uint64_t input_addr,
  uint64_t index_addr,
  uint64_t output_addr,
  uint32_t outer_size,
  int axis,
  uint32_t inner_size,
  int param_h
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return scatter_fp32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, param_addr, input_addr, index_addr,
        outer_size, axis, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return scatter_fp16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, param_addr, input_addr, index_addr,
        outer_size, axis, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return scatter_bf16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, param_addr, input_addr, index_addr,
        outer_size, axis, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return scatter_int32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, param_addr, input_addr, index_addr,
        outer_size, axis, inner_size, param_h
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
	TORCH_CHECK(false, "Scatter failed!");
}

template <typename scalar_t>
static void scatter_add_impl(
  uint64_t output_addr,
  uint64_t src_addr,
  uint64_t index_addr,
  uint32_t outer_size,
  uint32_t inner_size,
  int param_h
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return scatter_add_fp32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, src_addr, index_addr,
        outer_size, inner_size, param_h
        );
    }  else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return scatter_add_fp16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, src_addr, index_addr,
        outer_size, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return scatter_add_bf16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, src_addr, index_addr,
        outer_size, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return scatter_add_int32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, src_addr, index_addr,
        outer_size, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      return scatter_add_int16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, src_addr, index_addr,
        outer_size, inner_size, param_h
        );
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      return scatter_add_int8(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, src_addr, index_addr,
        outer_size, inner_size, param_h
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
	TORCH_CHECK(false, "Scatter add failed!");
}

#endif

namespace at {
Tensor &slice_scatter_out_tpu(const Tensor &self, const Tensor &src,
                              int64_t dim, c10::optional<int64_t> start,
                              c10::optional<int64_t> end, int64_t step,
                              Tensor &out) {
  TIMING_START;
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
#else
  int start_value = start.has_value() ? start.value() : 0;
  int end_value = end.has_value() ? end.value() : self.size(dim);
  int num_c = 1;
  for (int i = 0; i < dim; i++) {
    num_c *= self.size(i);
  }

  Tensor indices = arange(start_value, end_value, step, at::kInt)
                       .unsqueeze(0)
                       .unsqueeze(1)
                       .unsqueeze(-1)
                       .expand({1, num_c, -1, 1})
                       .to(self.device());
#ifdef USING_PPL
    if (usePPLKernels())
    {
  uint32_t inner_size = 1;
  for (const auto i : c10::irange((dim + 1), self.dim())) {
    inner_size *= self.size(i);
  }

  AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "slice_scatter", [&] {
        slice_scatter_impl<scalar_t>(
              reinterpret_cast<uint64_t>(src.data_ptr()),
              reinterpret_cast<uint64_t>(self.data_ptr()),
              reinterpret_cast<uint64_t>(indices.data_ptr()),
              reinterpret_cast<uint64_t>(out.data_ptr()),
              num_c,
              self.size(dim),
              inner_size,
              src.size(dim)
            );
      });
    } else
#endif
    {
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnSliceScatterAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, src),
      tpu::TPUGenerateTpudnnTensor(stream, indices),
      dim,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}

Tensor slice_scatter_tpu(const Tensor &self, const Tensor &src, int64_t dim,
                         c10::optional<int64_t> start,
                         c10::optional<int64_t> end, int64_t step) {
  Tensor out = empty_like(self);
  out = slice_scatter_out_tpu(self, src, dim, start, end, step, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("slice_scatter", slice_scatter_tpu);
  // m.impl("slice_scatter.out", slice_scatter_out_tpu);
}

Tensor &scatter_add_inplace_tpu(Tensor &self, int64_t dim, const Tensor &index,
                       const Tensor &src) {
  TIMING_START;
#ifdef USING_PPL
  if (usePPLKernels())
  {

  uint32_t outer_size = 1;
  uint32_t inner_size = 1;
  for (const auto i : c10::irange(dim)) {
    outer_size *= self.size(i);
  }
  for (const auto i : c10::irange((dim), self.dim())) {
    inner_size *= self.size(i);
  }

  AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "scatter_add", [&] {
        scatter_add_impl<scalar_t>(
              reinterpret_cast<uint64_t>(self.data_ptr()),
              reinterpret_cast<uint64_t>(src.data_ptr()),
              reinterpret_cast<uint64_t>(index.data_ptr()),
              outer_size, inner_size, index.size(dim)
            );
      });

  } else
#endif
  {
  int inplace_add = 1;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnScatterAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, src),
      tpu::TPUGenerateTpudnnTensor(stream, index),
      dim,
      inplace_add);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }

  TIMING_END;
  SHOW_TENSOR_OP(self, self);
  return self;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("scatter_add_", scatter_add_inplace_tpu);
}

Tensor& scatter_src_inplace_tpu(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src)
{
  TIMING_START;
  int inplace_add = 0;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnScatterAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, src),
      tpu::TPUGenerateTpudnnTensor(stream, index),
      dim,
      inplace_add);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  return self;
}

Tensor scatter_src_tpu(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src)
{
  TIMING_START;
  at::Tensor out = self.clone();
  int inplace_add = 0;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnScatterAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, out),
      tpu::TPUGenerateTpudnnTensor(stream, src),
      tpu::TPUGenerateTpudnnTensor(stream, index),
      dim,
      inplace_add);
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("scatter_.src", scatter_src_inplace_tpu); // inplace
  m.impl("scatter.src", scatter_src_tpu); // out-of-place
}
}  // namespace at
