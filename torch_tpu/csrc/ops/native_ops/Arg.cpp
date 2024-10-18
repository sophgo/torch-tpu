#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

#define ARGMAX_MODE 0
#define ARGMIN_MODE 1
#define MAX_DIM_MODE 2
#define MIN_DIM_MODE 3

namespace at {
Tensor &argmax_out_tpu(const Tensor &self, c10::optional<int64_t> dim,
                       bool keepdim, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(out);
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = argmax( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
  if ( self.dtype() == caffe2::TypeMeta::Make<long>() ||
       self.dtype() == caffe2::TypeMeta::Make<int>() )
  {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = argmax( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
    TIMING_END(tpu::CPU_LAYER);
    return out;
  }

  if (dim.has_value()) {
    if (dim.value() < 0) {
      dim = dim.value() + self.dim();
    }
    TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
  }
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor values = empty({out.sizes()}, options);
  Tensor buffer = empty({out.sizes()}, options);
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim.has_value() ? dim.value() : self.dim(), ARGMAX_MODE,
      tpu::TPUGenerateTpudnnTensor(stream, values),
      tpu::TPUGenerateTpudnnTensor(stream, out),
      tpu::TPUGenerateTpudnnTensor(stream, buffer));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::ARGMAX);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("argmax.out", argmax_out_tpu); }

Tensor &argmin_out_tpu(const Tensor &self, c10::optional<int64_t> dim,
                       bool keepdim, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(out);
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = argmin( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
  if (dim.has_value()) {
    if (dim.value() < 0) {
      dim = dim.value() + self.dim();
    }
    TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
  }
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor values = empty({out.sizes()}, options);
  Tensor buffer = empty({out.sizes()}, options);
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim.has_value() ? dim.value() : self.dim(), ARGMIN_MODE,
      tpu::TPUGenerateTpudnnTensor(stream, values),
      tpu::TPUGenerateTpudnnTensor(stream, out),
      tpu::TPUGenerateTpudnnTensor(stream, buffer));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::ARGMIN);
#endif
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("argmin.out", argmin_out_tpu); }

std::tuple<Tensor &, Tensor &> max_dim_max_out_tpu(const Tensor &self,
                                                   int64_t dim, bool keepdim,
                                                   Tensor &values,
                                                   Tensor &indices) {
  CHECK_TENSOR_IN_DEVICE(indices);
  CHECK_TENSOR_IN_DEVICE(values);
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu =  max(self.cpu(),dim,keepdim);
    values = TENSOR_TO_TPU(std::get<0>(out_cpu));
    indices = TENSOR_TO_TPU(std::get<1>(out_cpu));
#else
  if (dim < 0) {
    dim = dim + self.dim();
  }
  TORCH_CHECK(dim >= 0 || dim < self.dim());

  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor buffer = empty({indices.sizes()}, options);
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim, 
      MAX_DIM_MODE,
      tpu::TPUGenerateTpudnnTensor(stream, values),
      tpu::TPUGenerateTpudnnTensor(stream, indices),
      tpu::TPUGenerateTpudnnTensor(stream, buffer));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::MAX_DIM);
#endif
  return {values, indices};
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("max.dim_max", max_dim_max_out_tpu); }

std::tuple<Tensor &, Tensor &> min_dim_min_out_tpu(const Tensor &self,
                                                   int64_t dim, bool keepdim,
                                                   Tensor &values,
                                                   Tensor &indices) {
  CHECK_TENSOR_IN_DEVICE(indices);
  CHECK_TENSOR_IN_DEVICE(values);
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu =  min(self.cpu(),dim,keepdim);
    values = TENSOR_TO_TPU(std::get<0>(out_cpu));
    indices = TENSOR_TO_TPU(std::get<1>(out_cpu));
#else
  if (dim < 0) {
    dim = dim + self.dim();
  }
  TORCH_CHECK(dim >= 0 || dim < self.dim());
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor buffer = empty({indices.sizes()}, options);
  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim, 
      MIN_DIM_MODE,
      tpu::TPUGenerateTpudnnTensor(stream, values),
      tpu::TPUGenerateTpudnnTensor(stream, indices),
      tpu::TPUGenerateTpudnnTensor(stream, buffer));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::MIN_DIM);
#endif
  return {values, indices};
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("min.dim_min", min_dim_min_out_tpu); }

}  // namespace at