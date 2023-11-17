#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUDeviceManager.h"
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

  if (self.dim() == 0) {
    auto out_cpu = topk(self.cpu(), k, axis, largest, sorted);
    tpu::TPUCopyHostToDevice(values.data_ptr(),
                             std::get<0>(out_cpu).contiguous().data_ptr(),
                             values.nbytes());
    tpu::TPUCopyHostToDevice(indices.data_ptr(),
                             std::get<1>(out_cpu).contiguous().data_ptr(),
                             indices.nbytes());
  } else {
    if (axis < 0) {
      axis += self.dim();
    }
    TORCH_CHECK(axis <= self.dim())
    TORCH_CHECK(k <= self.size(axis));

    Tensor self_temp = self;
    Tensor values_temp = values;
    Tensor indices_temp = indices.to(torch::kInt32);
    if (self.dtype() == caffe2::TypeMeta::Make<at::Half>() ||
        self.dtype() == caffe2::TypeMeta::Make<at::BFloat16>()) {
      self_temp = self_temp.to(torch::kFloat);
      values_temp = values_temp.to(torch::kFloat);
    }

    TIMING_START
    bm_status_t status = sgdnnTopk(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self_temp), k,
        axis, largest, sorted, tpu::TPUGenerateSgdnnTensor(values_temp),
        tpu::TPUGenerateSgdnnTensor(indices_temp));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::TOPK)
    tpu::TPUCopyDeviceToDevice(values.data_ptr(),
                               values_temp.to(values.dtype()).data_ptr(),
                               values.nbytes());
    tpu::TPUCopyDeviceToDevice(indices.data_ptr(),
                               indices_temp.to(indices.dtype()).data_ptr(),
                               indices.nbytes());
  }
  SHOW_TENSOR_OP(self, values, indices);
  return {values, indices};
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("topk.values", topk_values_tpu); }

std::tuple<Tensor &, Tensor &> sort_values_stable_tpu(const Tensor &self, c10::optional<bool> stable,
                                                      int64_t axis, bool descending, Tensor &values,
                                                      Tensor &indices) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(values);
  CHECK_TENSOR_IN_DEVICE(indices);

  if (self.dim() == 0) {
    auto out_cpu = sort(self.cpu(), stable, axis, descending);
    tpu::TPUCopyHostToDevice(values.data_ptr(),
                             std::get<0>(out_cpu).contiguous().data_ptr(),
                             values.nbytes());
    tpu::TPUCopyHostToDevice(indices.data_ptr(),
                             std::get<1>(out_cpu).contiguous().data_ptr(),
                             indices.nbytes());
  } else {
    if (axis < 0) {
      axis += self.dim();
    }
    TORCH_CHECK(axis <= self.dim())

    Tensor self_temp = self;
    Tensor values_temp = values;
    Tensor indices_temp = indices.to(torch::kInt32);
    if (self.dtype() == caffe2::TypeMeta::Make<at::Half>() ||
        self.dtype() == caffe2::TypeMeta::Make<at::BFloat16>()) {
      self_temp = self_temp.to(torch::kFloat);
      values_temp = values_temp.to(torch::kFloat);
    }

    TIMING_START
    bm_status_t status =
        sgdnnTopk(tpu::TPUGetDeviceHandle(),
                  tpu::TPUGenerateSgdnnTensor(self_temp), self.size(axis), axis,
                  descending, false, tpu::TPUGenerateSgdnnTensor(values_temp),
                  tpu::TPUGenerateSgdnnTensor(indices_temp));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::TOPK)
    tpu::TPUCopyDeviceToDevice(values.data_ptr(),
                               values_temp.to(values.dtype()).data_ptr(),
                               values.nbytes());
    tpu::TPUCopyDeviceToDevice(indices.data_ptr(),
                               indices_temp.to(indices.dtype()).data_ptr(),
                               indices.nbytes());
  }

  return {values, indices};
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sort.values_stable", sort_values_stable_tpu);
}

} // namespace at