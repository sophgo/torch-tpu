#include <torch/torch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>

#include "TPUDeviceManager.h"
#include "TPUTorchUtils.h"
#include "sgdnn_api.h"
#include "common/config.h"

namespace at {

/*
 * k:       需要返回的最大值的个数，即取 top-k 操作中的 k 值。
 * dim:     可选参数，用于指定在哪个维度上进行 top-k 操作。如果不指定，
 *          则默认为对整个张量进行 top-k 操作。
 * largest: 可选参数，用于指定是返回最大的 k 个值还是最小的 k 个值。
 *          默认为 True，表示返回最大的 k 个值。
 * sorted:  可选参数，用于指定返回的结果是否按照降序排列。默认为 True，表示按照降序排列。
 * values:  取出k个的top值
 * indices：top值在其维度中的对应索引
 */
std::tuple<Tensor&, Tensor&> topk_values_tpu(const Tensor &self, int64_t k, 
        int64_t dim, bool largest, bool sorted, Tensor &values, Tensor &indices) {
    if(self.dim() > 0) {
        CHECK_TENSOR_IN_DEVICE(self);
    }
    CHECK_TENSOR_IN_DEVICE(values);
    CHECK_TENSOR_IN_DEVICE(indices);
    if(dim >= 0) {
        TORCH_CHECK(dim <= self.dim())
        TORCH_CHECK(k <= self.size(dim));
    }
    else {
        TORCH_CHECK(dim >= (1 - self.dim()));
        TORCH_CHECK(k <= self.size(self.dim() + dim));
    }

    if(self.dim() == 0) {
        auto out_cpu = topk(self, k, dim, largest, sorted);
        tpu::TPUCopyHostToDevice(values.data_ptr(),
                                 std::get<0>(out_cpu).contiguous().data_ptr(), values.nbytes());
        tpu::TPUCopyHostToDevice(indices.data_ptr(),
                                 std::get<1>(out_cpu).contiguous().data_ptr(), indices.nbytes());
    }
    else {

TIMING_START
        bm_status_t status = sgdnnTopk(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                                       k, dim, largest, sorted, tpu::TPUGenerateSgdnnTensor(values),
                                       tpu::TPUGenerateSgdnnTensor(indices));
        TORCH_CHECK(status == BM_SUCCESS);
TIMING_END(tpu::TOPK)

    }

    return {values, indices};
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("topk.values", topk_values_tpu);
}

}   // namespace at