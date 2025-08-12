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
  TIMING_START;
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
    auto out_cpu = argmax( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
    return out;
  }

  if (dim.has_value()) {
    if (dim.value() < 0) {
      dim = dim.value() + self.dim();
    }
    TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
  }
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim.has_value() ? dim.value() : self.dim(), ARGMAX_MODE,
      tpudnnUndefinedTensor(),
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}

Tensor argmax_tpu(const Tensor &self, c10::optional<int64_t> dim, bool keepdim) {
  CHECK_TENSOR_IN_DEVICE(self);

  if (dim.has_value()) {
    if (dim.value() < 0) {
      dim = dim.value() + self.dim();
    }
    TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
  }
  TensorOptions options = TensorOptions ( self.device() ).dtype ( torch::kInt32 );
  std::vector<int64_t> sizes_vec;
  if ( keepdim )
  {
    if ( dim.has_value() )
    {
      for ( int i = 0; i < self.dim(); i++ ) { sizes_vec.push_back( self.size(i) ); }
      sizes_vec[dim.value()] = 1;
    }
    else
    {
      for ( int i = 0; i < self.dim(); i++ ) { sizes_vec.push_back( 1 ); }
    }
  }
  else
  {
    if (dim.has_value())
    {
      for (int i = 0; i < dim.value(); i++) { sizes_vec.push_back( self.size(i) ); }
      for (int i = dim.value() + 1; i < self.dim(); i++) { sizes_vec.push_back( self.size(i) ); }
    }
    else
    {
      // sizes_vec.push_back( 1 );
    }
  }
  IntArrayRef sizes(sizes_vec.data(), sizes_vec.size());
  Tensor out = empty(sizes, options);
  out = argmax_out_tpu(self, dim, keepdim, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("argmax", argmax_tpu); }
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("argmax.out", argmax_out_tpu); }

Tensor &argmin_out_tpu(const Tensor &self, c10::optional<int64_t> dim,
                       bool keepdim, Tensor &out) {
  TIMING_START;
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
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim.has_value() ? dim.value() : self.dim(), ARGMIN_MODE,
      tpudnnUndefinedTensor(),
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
#endif
  return out;
}

Tensor argmin_tpu(const Tensor &self, c10::optional<int64_t> dim, bool keepdim) {
  CHECK_TENSOR_IN_DEVICE(self);

  if (dim.has_value()) {
    if (dim.value() < 0) {
      dim = dim.value() + self.dim();
    }
    TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
  }
  TensorOptions options = TensorOptions ( self.device() ).dtype ( torch::kInt32 );
  std::vector<int64_t> sizes_vec;
  if ( keepdim )
  {
    if ( dim.has_value() )
    {
      for ( int i = 0; i < self.dim(); i++ ) { sizes_vec.push_back( self.size(i) ); }
      sizes_vec[dim.value()] = 1;
    }
    else
    {
      for ( int i = 0; i < self.dim(); i++ ) { sizes_vec.push_back( 1 ); }
    }
  }
  else
  {
    if (dim.has_value())
    {
      for (int i = 0; i < dim.value(); i++) { sizes_vec.push_back( self.size(i) ); }
      for (int i = dim.value() + 1; i < self.dim(); i++) { sizes_vec.push_back( self.size(i) ); }
    }
    else
    {
      // sizes_vec.push_back( 1 );
    }
  }
  IntArrayRef sizes(sizes_vec.data(), sizes_vec.size());
  Tensor out = empty(sizes, options);
  out = argmin_out_tpu(self, dim, keepdim, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("argmin", argmin_tpu); }
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("argmin.out", argmin_out_tpu); }

std::tuple<Tensor &, Tensor &> max_dim_max_out_tpu(const Tensor &self,
                                                   int64_t dim, bool keepdim,
                                                   Tensor &values,
                                                   Tensor &indices) {
  TIMING_START;
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

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim, 
      MAX_DIM_MODE,
      tpu::TPUGenerateTpudnnTensor(stream, values),
      tpu::TPUGenerateTpudnnTensor(stream, indices));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  return {values, indices};
}
std::tuple<Tensor, Tensor> max_dim_tpu(const Tensor &self,
                                       int64_t dim, bool keepdim)
{
  CHECK_TENSOR_IN_DEVICE(self);
  if (dim < 0) { dim = dim + self.dim();}
  TORCH_CHECK(dim >= 0 || dim < self.dim());
  std::vector<int64_t> sizes_vec;
  if ( keepdim )
  {
    for ( int i = 0; i < self.dim(); i++ ) { sizes_vec.push_back( self.size(i) ); }
    sizes_vec[dim] = 1;
  }
  else
  {
    for (int i = 0; i < dim; i++) { sizes_vec.push_back( self.size(i) ); }
    for (int i = dim + 1; i < self.dim(); i++) { sizes_vec.push_back( self.size(i) ); }
  }
  IntArrayRef sizes(sizes_vec.data(), sizes_vec.size());
  TensorOptions idx_options = TensorOptions ( self.device() ).dtype ( torch::kInt32 );
  TensorOptions val_options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor idx    = empty(sizes, idx_options);
  Tensor values = empty(sizes, val_options); 
  max_dim_max_out_tpu(self, dim, keepdim, values, idx);
  return {values, idx};
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("max.dim", max_dim_tpu); }
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("max.dim_max", max_dim_max_out_tpu); }

std::tuple<Tensor &, Tensor &> min_dim_min_out_tpu(const Tensor &self,
                                                   int64_t dim, bool keepdim,
                                                   Tensor &values,
                                                   Tensor &indices) {
  TIMING_START;
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

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnArgAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      dim, 
      MIN_DIM_MODE,
      tpu::TPUGenerateTpudnnTensor(stream, values),
      tpu::TPUGenerateTpudnnTensor(stream, indices));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  return {values, indices};
}

std::tuple<Tensor, Tensor> min_dim_tpu(const Tensor &self,
                                       int64_t dim, bool keepdim)
{
  CHECK_TENSOR_IN_DEVICE(self);
  if (dim < 0) { dim = dim + self.dim();}
  TORCH_CHECK(dim >= 0 || dim < self.dim());
  std::vector<int64_t> sizes_vec;
  if ( keepdim )
  {
    for ( int i = 0; i < self.dim(); i++ ) { sizes_vec.push_back( self.size(i) ); }
    sizes_vec[dim] = 1;
  }
  else
  {
    for (int i = 0; i < dim; i++) { sizes_vec.push_back( self.size(i) ); }
    for (int i = dim + 1; i < self.dim(); i++) { sizes_vec.push_back( self.size(i) ); }
  }
  IntArrayRef sizes(sizes_vec.data(), sizes_vec.size());
  TensorOptions idx_options = TensorOptions ( self.device() ).dtype ( torch::kInt32 );
  TensorOptions val_options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor idx    = empty(sizes, idx_options);
  Tensor values = empty(sizes, val_options); 
  min_dim_min_out_tpu(self, dim, keepdim, values, idx);
  return {values, idx};
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("min.dim", min_dim_tpu); }
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("min.dim_min", min_dim_min_out_tpu); }

}  // namespace at