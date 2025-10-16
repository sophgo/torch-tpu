#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at{
/**
 * * * * * * * * * * * * * * * * * * * cumax * * * * * * * * * * * * * * * * * * 
 */
// called by https://pytorch.org/docs/2.1/generated/torch.cummax.html#torch.cummax
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
    return cummax_out_tpu(self, dim, values, indices);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("cummax.out", cummax_out_tpu);
    m.impl("cummax",     cummax_tpu);
}

/**
 * * * * * * * * * * * * * * * * * * * cummin * * * * * * * * * * * * * * * * * * 
 */
// called by https://pytorch.org/docs/2.1/generated/torch.cummin.html#torch.cummin
std::tuple<Tensor &,Tensor &> cummin_out_tpu(const Tensor & self, int64_t dim, 
                                             Tensor & values, Tensor & indices) {
    CPU_IMPL_WARNING();
    auto outputs_cpu = torch::cummin(self.cpu(), dim);
    values  = (std::get<0> (outputs_cpu)).to(values.device());
    indices = (std::get<1> (outputs_cpu)).to(indices.device());
    return {values, indices};
}

std::tuple<Tensor,Tensor> cummin_tpu(const Tensor & self, int64_t dim) {
    TensorOptions values_opts  = self.options();
    TensorOptions indices_opts = self.options().dtype(ScalarType::Int);
    auto values  = empty(self.sizes(), values_opts);
    auto indices = empty(self.sizes(), indices_opts); 
    return cummin_out_tpu(self, dim, values, indices);
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("cummin.out", cummin_out_tpu);
    m.impl("cummin",     cummin_tpu);
}

/**
 * * * * * * * * * * * * * * * * * * cumprod * * * * * * * * * * * * * * * * * * 
 */
// called by https://pytorch.org/docs/2.1/generated/torch.cumprod.html#torch.cumprod
Tensor & cumprod_out_tpu(const Tensor & self, int64_t dim,
                        c10::optional<ScalarType> dtype, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::cumprod(self.cpu(), dim, dtype);
    out = out_cpu.to(out.device());
    return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("cumprod.out", cumprod_out_tpu);
}
/**
 * * * * * * * * * * * * * * * * * * * cumsum * * * * * * * * * * * * * * * * * * 
 */
// called by https://pytorch.org/docs/2.1/generated/torch.cumsum.html#torch.cumsum
Tensor & cumsum_out_tpu(const Tensor & self, int64_t dim, 
                        c10::optional<ScalarType> dtype, Tensor & out) {
    ScalarType desired_dtype = dtype.has_value() ? *dtype : out.scalar_type();
    TORCH_CHECK(desired_dtype == out.scalar_type());
                            
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnCumsumAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, out),
        dim
    );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    return out;
}
Tensor cumsum_tpu(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype=c10::nullopt) {
    ScalarType desired_dtype = dtype.has_value() ? *dtype : self.scalar_type();
    auto out  = empty(self.sizes(), self.options().dtype(desired_dtype));
    return cumsum_out_tpu(self, dim, c10::optional<ScalarType>(desired_dtype), out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("cumsum.out", cumsum_out_tpu);
    m.impl("cumsum",     cumsum_tpu);
}

// called by https://pytorch.org/docs/2.1/generated/torch.logcumsumexp.html#torch.logcumsumexp
Tensor & _logcumsumexp_out_tpu(const Tensor & self, int64_t dim, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::logcumsumexp(self.cpu(), dim);
    out = out_cpu.to(out.device());
    return out;
}
Tensor _logcumsumexp_tpu(const Tensor & self, int64_t dim) {
    auto out = empty(self.sizes(), self.options());
    return _logcumsumexp_out_tpu(self, dim, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("_logcumsumexp.out", _logcumsumexp_out_tpu);
    m.impl("_logcumsumexp",     _logcumsumexp_tpu);
}
}; // namespace at