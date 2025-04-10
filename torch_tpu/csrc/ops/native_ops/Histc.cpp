#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

//called by https://pytorch.org/docs/2.1/generated/torch.histc.html#torch.histc
namespace at {
Tensor & histc_out_tpu(const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::histc(self.cpu(), bins, min, max);
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
    return out;
}

Tensor histc_tpu(const Tensor & self, int64_t bins, const Scalar & min, const Scalar & max) {
    TensorOptions options = self.options().dtype(ScalarType::Float);
    auto out = empty({bins}, options);
    return histc_out_tpu(self, bins, min, max, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("histc.out", histc_out_tpu);
    m.impl("histc",     histc_tpu);
}

// called by https://pytorch.org/docs/2.1/generated/torch.histogram.html#torch.histogram
std::tuple<Tensor &,Tensor &> histogram_bins_t_out_tpu(const Tensor & self, const Tensor & bins, 
    const c10::optional<Tensor> & weight, bool density, Tensor & hist, Tensor & bin_edges) {
    CPU_IMPL_WARNING();
    auto outs_cpu = torch::histogram(self, bins, weight, density);
    hist      = std::get<0>(outs_cpu).to(hist.device());
    bin_edges = std::get<1>(outs_cpu).to(bin_edges.device());
    return {hist, bin_edges};
}
std::tuple<Tensor &,Tensor &> histogram_bin_ct_out_tpu(const Tensor & self, int64_t bins, 
    c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, 
    bool density, Tensor & hist, Tensor & bin_edges) {
    CPU_IMPL_WARNING();
    auto outs_cpu = torch::histogram(self, bins, range, weight, density);
    hist      = std::get<0>(outs_cpu).to(hist.device());
    bin_edges = std::get<1>(outs_cpu).to(bin_edges.device());
    return {hist, bin_edges};
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("histogram.bins_tensor_out", histogram_bins_t_out_tpu);
    m.impl("histogram.bin_ct_out",      histogram_bin_ct_out_tpu);
}
// called by https://pytorch.org/docs/2.1/generated/torch.histogramdd.html#torch.histogramdd
void _histogramdd_bin_edges_out_tpu(const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range,
                const c10::optional<Tensor> & weight, bool density, TensorList out) {
    CPU_IMPL_WARNING();
    auto outs_cpu = torch::_histogramdd_bin_edges(self.cpu(), bins, range, 
                        weight.has_value() ?  c10::optional<Tensor>(weight.value().cpu()) : c10::optional<Tensor>(),
                        density);
    for (size_t i =0; i < outs_cpu.size(); i++){ 
        tpu::TPUCopyHostToDevice ( out[i].data_ptr(), outs_cpu[i].contiguous().data_ptr(), out[i].nbytes() );
    }
    return ; 
}
std::vector<Tensor> _histogramdd_bin_edges_tpu(const Tensor & self, IntArrayRef bins,
            c10::optional<ArrayRef<double>> range, const c10::optional<Tensor> & weight, bool density) {
    CPU_IMPL_WARNING();
    std::vector<Tensor> out;
    auto outs_cpu = torch::_histogramdd_bin_edges(self.cpu(), bins, range, 
                        weight.has_value() ?  c10::optional<Tensor>(weight.value().cpu()) : c10::optional<Tensor>(),
                        density);    
    for (size_t i = 0; i < outs_cpu.size(); i++) {
        auto oi = outs_cpu[i].to(self.device());
        out.push_back(oi);
    }
    return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("_histogramdd_bin_edges.out", _histogramdd_bin_edges_out_tpu);
    m.impl("_histogramdd_bin_edges",     _histogramdd_bin_edges_tpu);
}

Tensor & _histogramdd_from_bin_cts_out_tpu(const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range,
        const c10::optional<Tensor> & weight, bool density, Tensor & out) {
    CPU_IMPL_WARNING();
    auto out_cpu = torch::_histogramdd_from_bin_cts(self.cpu(), bins, range,
                        weight.has_value() ?  c10::optional<Tensor>(weight.value().cpu()) : c10::optional<Tensor>(),
                        density);
    out = out_cpu.to(self.device());
    return out;
}
Tensor _histogramdd_from_bin_cts_tpu(const Tensor & self, IntArrayRef bins, c10::optional<ArrayRef<double>> range,
        const c10::optional<Tensor> & weight, bool density) {
    Tensor out = Tensor();
    return _histogramdd_from_bin_cts_out_tpu(self, bins, range, weight, density, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl("_histogramdd_from_bin_cts.out", _histogramdd_from_bin_cts_out_tpu);
    m.impl("_histogramdd_from_bin_cts",     _histogramdd_from_bin_cts_tpu);
}
}