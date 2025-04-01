#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"
#include "sgdnn_api.h"

namespace at {

/*
 * k:       需要返回的最大值的个数，即取 top-k 操作中的 k 值。
 * dim:     可选参数，用于指定在哪个维度上进行 top-k 操作。如果不指定，
 *          则默认为对整个张量进行 top-k 操作。
 * largest: 可选参数，用于指定是返回最大的 k 个值还是最小的 k 个值。
 *          默认为 True，表示返回最大的 k 个值。
 * sorted:  可选参数，用于指定返回的结果是否按照降序排列。默认为
 * True，表示按照降序排列。 values:  取出k个的top值
 * indices：top值在其维度中的对应索引
 */
std::tuple<Tensor &, Tensor &> topk_values_tpu(const Tensor &self, int64_t k,
                                               int64_t axis, bool largest,
                                               bool sorted, Tensor &values,
                                               Tensor &indices) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(values);
  CHECK_TENSOR_IN_DEVICE(indices);
  TORCH_CHECK( indices.dtype() == caffe2::TypeMeta::Make<int>(), "indeices should be int32 dtype" );

  if (axis < 0) {
    axis += self.dim();
  }

  TORCH_CHECK(axis <= self.dim())
  TORCH_CHECK(k <= self.size(axis));

  if (self.size(axis) <= 256) {
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnTopkAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        k,
        axis,
        largest,
        sorted,
        tpu::TPUGenerateTpudnnTensor(stream, values),
        tpu::TPUGenerateTpudnnTensor(stream, indices)
    );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  } else {
    Tensor self_temp = self;
    Tensor values_temp = values;
    if (self.dtype() == caffe2::TypeMeta::Make<Half>() ||
        self.dtype() == caffe2::TypeMeta::Make<BFloat16>()) {
      self_temp = self_temp.to(torch::kFloat);
      values_temp = values_temp.to(torch::kFloat);
    }

    TIMING_START;
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnTopkAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self_temp),
        k,
        axis,
        largest,
        sorted,
        tpu::TPUGenerateTpudnnTensor(stream, values_temp),
        tpu::TPUGenerateTpudnnTensor(stream, indices)
    );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);

    if (self.dtype() == caffe2::TypeMeta::Make<Half>() ||
        self.dtype() == caffe2::TypeMeta::Make<BFloat16>()) {
      values = values_temp.to(values.dtype());
    }
  }
  TIMING_END(tpu::TOPK);

  SHOW_TENSOR_OP(self, values, indices);
  return {values, indices};
}

std::tuple<Tensor,Tensor>
topk_tpu(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  TensorOptions val_options     = self.options();
  TensorOptions indices_options = self.options().dtype(ScalarType::Int);
  int64_t dim_ = dim < 0 ? dim + self.dim() : dim;
  std::vector<int64_t> sizes_vec;
  for (int i =0; i < self.dim(); i++)
  {
    if ( i == dim_ ) { sizes_vec.push_back(k); }
    else { sizes_vec.push_back( self.size(i) ); }
  }
  IntArrayRef sizes(sizes_vec.data(), sizes_vec.size());
  auto values  = empty(sizes, val_options);
  auto indices = empty(sizes, indices_options);
  topk_values_tpu(self, k, dim, largest, sorted, values, indices);
  return std::tuple<Tensor,Tensor>(values, indices);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("topk.values", topk_values_tpu);
  m.impl("topk", topk_tpu);
}
// =========== sort
std::tuple<Tensor &, Tensor &>
sort_values_stable_tpu(const Tensor &self, c10::optional<bool> stable,
                       int64_t axis, bool descending, Tensor &values,
                       Tensor &indices) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(values);
  CHECK_TENSOR_IN_DEVICE(indices);
  TORCH_CHECK( indices.dtype() == caffe2::TypeMeta::Make<int>(), "indeices should be int32 dtype" );

  if (axis < 0) { axis += self.dim(); }
  TORCH_CHECK(axis <= self.dim())

  Tensor self_temp = self;
  Tensor values_temp = values;
  if (self.dtype() == caffe2::TypeMeta::Make<Half>() ||
      self.dtype() == caffe2::TypeMeta::Make<BFloat16>()) {
    self_temp = self_temp.to(torch::kFloat);
    values_temp = values_temp.to(torch::kFloat);
  }

  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnTopkAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self_temp),
      self.size(axis),
      axis,
      descending,
      false,
      tpu::TPUGenerateTpudnnTensor(stream, values_temp),
      tpu::TPUGenerateTpudnnTensor(stream, indices)
  );
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);

  if (self.dtype() == caffe2::TypeMeta::Make<Half>() ||
      self.dtype() == caffe2::TypeMeta::Make<BFloat16>()) {
    values = values_temp.to(values.dtype());
  }
  TIMING_END(tpu::TOPK);

  return {values, indices};
}

std::tuple<Tensor &,Tensor &>
sort_values_tpu(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  return sort_values_stable_tpu(self, c10::nullopt, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor>
sort_stable_tpu(const Tensor & self, c10::optional<bool> stable,
                int64_t dim, bool descending)
{
  TORCH_CHECK(!stable.has_value(), "[OP] sort not support [ARG]-stable");
  TensorOptions val_options     = self.options();
  TensorOptions indices_options = self.options().dtype(ScalarType::Int);
  auto values  = empty(self.sizes(), val_options);
  auto indices = empty(self.sizes(), indices_options);
  sort_values_stable_tpu(self, stable, dim, descending, values, indices);
  return {values, indices};
}

std::tuple<Tensor,Tensor>
sort_tpu(const Tensor & self, int64_t dim, bool descending) {
  TensorOptions val_options     = self.options();
  TensorOptions indices_options = self.options().dtype(ScalarType::Int);
  auto values  = empty(self.sizes(), val_options);
  auto indices = empty(self.sizes(), indices_options);
  sort_values_stable_tpu(self, c10::nullopt, dim, descending, values, indices);
  return {values, indices};
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sort.values_stable", sort_values_stable_tpu);
  m.impl("sort.values", sort_values_tpu);
  m.impl("sort.stable", sort_stable_tpu);
  m.impl("sort", sort_tpu);
}

// =========== argsort
Tensor & argsort_stable_out_tpu(const Tensor & self, bool stable, int64_t dim, bool descending, Tensor & out) {
  TORCH_CHECK(false, "should not be called now");
  return out;
}
Tensor argsort_stable_tpu(const Tensor & self, bool stable, int64_t dim, bool descending) {
  auto values_indices = sort_stable_tpu(self, c10::optional<bool>(stable), dim, descending);
  return std::get<1> ( values_indices );
}
Tensor argsort_tpu(const Tensor & self, int64_t dim, bool descending) {
  // values is no use, can be optimized.
  auto values_indices = sort_tpu(self, dim, descending);
  return std::get<1> ( values_indices );
}


TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("argsort.stable_out", argsort_stable_out_tpu);
  m.impl("argsort.stable", argsort_stable_tpu);
  m.impl("argsort", argsort_tpu);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("argsort", argsort_tpu);
}

} // namespace at
