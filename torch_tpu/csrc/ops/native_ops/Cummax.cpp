#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

// called by https://pytorch.org/docs/2.1/generated/torch.cummax.html#torch.cummax
namespace at{

std::tuple<Tensor &,Tensor &> cummax_out_tpu(const Tensor & self, int64_t dim, 
                                             Tensor & values, Tensor & indices) {
    CPU_IMPL_WARNING();
    auto outputs_cpu = torch::cummax(self.cpu(), dim);
    values  = (std::get<0> (outputs_cpu)).to(values.device());
    indices = (std::get<1> (outputs_cpu)).to(indices.device());
    return {values, indices};
}

std::tuple<Tensor,Tensor> cummax_tpu(const Tensor & self, int64_t dim) {
    TensorOptions values_opts  = self.options();
    TensorOptions indices_opts = self.options().dtype(ScalarType::Int);
    auto values  = empty(self.sizes(), values_opts);
    auto indices = empty(self.sizes(), indices_opts); 
    return _ops::cummax_out::call(self, dim, values, indices);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("cummax.out", cummax_out_tpu);
    m.impl("cummax",     cummax_tpu);
}
}; // namespace at