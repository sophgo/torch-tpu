#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
Tensor & norm_out_tpu ( const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, Tensor & out )
{
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = norm ( self.cpu(), p, dim, keepdim );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( p.has_value() )
  {
    TORCH_CHECK ( p.value().toDouble() == 2., "Only support 2-Norm now" );
  }
  // not support case, use cpu impl
  if (dim.size() != (size_t)self.dim() && dim.size() != 0 )
  {
    CPU_IMPL_WARNING();
    auto out_cpu = norm ( self.to(torch::kFloat).cpu(), p, dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
  }
  else
  {
    auto buffer = empty({8}, out.options().dtype(torch::kFloat32));
    TORCH_CHECK ( (int)dim.size() == self.dim() || dim.size() == 0 ); // TODO: Support partial dims
    for (int i = 0; i < (int)dim.size(); i++) { TORCH_CHECK ( dim[i] == i ); }
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnNorm2Async(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, buffer),
        keepdim,
        tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "norm.out", norm_out_tpu );
}

at::Tensor & linalg_vector_norm_out_tpu(
    const at::Tensor & self, const at::Scalar & scalar_ord, at::OptionalIntArrayRef opt_dim,
    bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)
{
  auto dim = opt_dim.value_or(IntArrayRef{});
  c10::optional<Scalar> ord = c10::optional<Scalar> (scalar_ord);
  return norm_out_tpu(self, ord, dim, keepdim, out);
}


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "linalg_vector_norm.out", linalg_vector_norm_out_tpu );
}

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4
std::vector<at::Tensor> _foreach_norm_tpu(at::TensorList tensors, const at::Scalar & ord, std::optional<c10::ScalarType>) {
#else
c10::ArrayRef<at::Tensor>, c10::Scalar const&, std::optional<c10::ScalarType>
#endif
  std::vector<at::Tensor> outs;
  for (const auto& t : tensors){
    outs.emplace_back(linalg_vector_norm(t, ord));
  }
  return outs;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_foreach_norm.Scalar", _foreach_norm_tpu );
}

} // namespace at
