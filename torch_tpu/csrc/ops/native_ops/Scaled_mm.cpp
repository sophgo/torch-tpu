#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"
// https://github.com/pytorch/pytorch/blob/bc47d539fc380f521dfcc25e895e46e6d5a1fd52/aten/src/ATen/native/cuda/Blas.cpp#L1127
namespace at
{
std::tuple<Tensor &,Tensor &> _scaled_mm_out_tpu(
        const Tensor & self,
        const Tensor & mat2,
        const c10::optional<Tensor> & bias,
        c10::optional<ScalarType> out_dtype,
        const c10::optional<Tensor> & scale_a,
        const c10::optional<Tensor> & scale_b,
        const c10::optional<Tensor> & scale_result,
        Tensor & out,
        Tensor & out_amax){
    CPU_IMPL_WARNING();
    // ERROR ===== cpu has no _scaled_mm.
    // auto outputs_cpu = torch::_scaled_mm(self.cpu(), mat2.cpu(),
    //                         bias.has_value() ? c10::optional<Tensor>(bias.value().cpu()) : c10::optional<Tensor>(),
    //                         out_dtype,
    //                         scale_a.has_value() ? c10::optional<Tensor>(scale_a.value().cpu()) : c10::optional<Tensor>(),
    //                         scale_b.has_value() ? c10::optional<Tensor>(scale_b.value().cpu()) : c10::optional<Tensor>(),
    //                         scale_result.has_value() ? c10::optional<Tensor>(scale_result.value().cpu()) : c10::optional<Tensor>());
    // out      = std::get<0> ( outputs_cpu ).to(out.device());
    // out_amax = std::get<1> ( outputs_cpu ).to(out.device());
    CHECK_TENSOR_IN_DEVICE( self );
    CHECK_TENSOR_IN_DEVICE( mat2 );
    if (bias.has_value())
      CHECK_TENSOR_IN_DEVICE(bias.value());
    if (scale_a.has_value())
      CHECK_TENSOR_IN_DEVICE(scale_a.value());
    if (scale_b.has_value())
      CHECK_TENSOR_IN_DEVICE(scale_b.value());
    if (scale_result.has_value())
      CHECK_TENSOR_IN_DEVICE(scale_result.value());
    CHECK_TENSOR_IN_DEVICE(out);
    CHECK_TENSOR_IN_DEVICE(out_amax);

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnMatmulFp8Async(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, mat2),
      bias.has_value()? tpu::TPUGenerateTpudnnTensor(stream, bias.value()) : tpudnnUndefinedTensor(),
      scale_a.has_value()? tpu::TPUGenerateTpudnnTensor(stream, scale_a.value()) : tpudnnUndefinedTensor(),
      scale_b.has_value()? tpu::TPUGenerateTpudnnTensor(stream, scale_b.value()) : tpudnnUndefinedTensor(),
      scale_result.has_value()? tpu::TPUGenerateTpudnnTensor(stream, scale_result.value()) : tpudnnUndefinedTensor(),
      tpu::TPUGenerateTpudnnTensor(stream, out),
      tpu::TPUGenerateTpudnnTensor(stream, out_amax)
    );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    return {out, out_amax};
}

std::tuple<Tensor,Tensor> _scaled_mm_tpu(
        const Tensor & self, const Tensor & mat2, const c10::optional<Tensor> & bias, c10::optional<ScalarType> out_dtype,
        const c10::optional<Tensor> & scale_a, const c10::optional<Tensor> & scale_b, const c10::optional<Tensor> & scale_result) {
    // Check sizes
    TORCH_CHECK(self.dim() == 2, "self must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");

    IntArrayRef mat1_sizes = self.sizes();
    IntArrayRef mat2_sizes = mat2.sizes();
    const auto out_dtype_ = out_dtype.value_or(self.scalar_type());
    Tensor out = empty({mat1_sizes[0], mat2_sizes[0]}, self.options().dtype(out_dtype_));
    Tensor amax = empty({}, self.options().dtype(ScalarType::Float));
    return _scaled_mm_out_tpu(self, mat2, bias, out_dtype, scale_a, scale_b, scale_result, out ,amax);
}


TORCH_LIBRARY_IMPL (aten, TPU, m)
{
  m.impl ("_scaled_mm.out", _scaled_mm_out_tpu);
  m.impl ("_scaled_mm",     _scaled_mm_tpu);
}

} // namespace at
