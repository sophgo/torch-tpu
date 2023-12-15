#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"

namespace at
{
Tensor & norm_out_tpu ( const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, Tensor & out )
{
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
  if (dim.size() != self.dim() && dim.size() != 0 )
  {
    CPU_IMPL_WARNING();
    auto out_cpu = norm ( self.to(torch::kFloat).cpu(), p, dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
  }
  else
  {
    TORCH_CHECK ( dim.size() == self.dim() || dim.size() == 0 ); // TODO: Support partial dims
    for (int i = 0; i < dim.size(); i++) { TORCH_CHECK ( dim[i] == i ); }
    TIMING_START;
    bm_status_t status = sgdnnNorm2 (
                        tpu::TPUGetDeviceHandle(),
                        tpu::TPUGenerateSgdnnTensor ( self ),
                        keepdim,
                        tpu::TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
    TIMING_END(tpu::NORM2);
  }
#endif
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

std::vector<at::Tensor> _foreach_norm_tpu(at::TensorList tensors, const at::Scalar & ord) {
  std::vector<at::Tensor> outs;
  for (const auto& t : tensors){
    IntArrayRef dim = IntArrayRef{};
    outs.emplace_back(norm(t, ord, dim, true));
  }
  return outs;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_foreach_norm.Scalar", _foreach_norm_tpu );
}

} // namespace at
