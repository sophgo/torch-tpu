#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.searchsorted.html#torch.searchsorted
namespace at {
Tensor & searchsorted_Tensor_out_tpu(const Tensor & sorted_sequence, const Tensor & self,
            bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::searchsorted(sorted_sequence.cpu(), self.cpu(), 1, right, side, sorter);
    out.copy_(out_cpu);
    return out;
}
Tensor searchsorted_Tensor_tpu(const Tensor & sorted_sequence, const Tensor & self, 
            bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter) {
    auto out = empty(self.sizes(), self.options().dtype(ScalarType::Int));
    return searchsorted_Tensor_out_tpu(sorted_sequence, self, out_int32, right, side, sorter, out);
}
Tensor & searchsorted_Scalar_out_tpu(const Tensor & sorted_sequence, const Scalar & self,
            bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::searchsorted(sorted_sequence.cpu(), self, 1, right, side, sorter);
    out.copy_(out_cpu);
    return out;
}
Tensor searchsorted_Scalar_tpu(const Tensor & sorted_sequence, const Scalar & self,
            bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<Tensor> & sorter) {
    auto out = empty({}, sorted_sequence.options().dtype(ScalarType::Int));
    return searchsorted_Scalar_out_tpu(sorted_sequence, self, out_int32, right, side, sorter, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("searchsorted.Tensor_out", searchsorted_Tensor_out_tpu);
    m.impl("searchsorted.Tensor",     searchsorted_Tensor_tpu);
    m.impl("searchsorted.Scalar_out", searchsorted_Scalar_out_tpu);
    m.impl("searchsorted.Scalar",     searchsorted_Scalar_tpu);
}
}