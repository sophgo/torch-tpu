#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at
{

static inline bool is_transposed ( const Tensor & tensor )
{
  if ( tensor.is_contiguous() )
  {
    return false;
  }
  if ( tensor.dim() == 2 )
  {
    return tensor.stride ( 0 ) == 1 && tensor.stride ( 1 ) == tensor.size ( 0 );
  }
  if ( tensor.dim() == 3 )
  {
    return tensor.stride ( 0 ) == tensor.size ( 1 ) * tensor.size ( 2 ) && tensor.stride ( 1 ) == 1 && tensor.stride ( 2 ) == tensor.size ( 1 );
  }
  return false;
}

Tensor & addmm_out_tpu ( const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out )
{
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( mat1 );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = addmm ( self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( ( alpha.toDouble() == 1. ) && ( beta.toDouble() == 1. ) && self.dim() == 1 )
  {
    auto mat1_ = mat1.is_contiguous() == false && is_transposed ( mat1 ) == false ? mat1.contiguous() : mat1;
    auto mat2_ = mat2.is_contiguous() == false && is_transposed ( mat2 ) == false ? mat2.contiguous() : mat2;


    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnMatmulAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, mat1_),
      tpu::TPUGenerateTpudnnTensor(stream, mat2_),
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );

    TIMING_END( tpu::ADDMM );
  }
  else
  {
    TORCH_CHECK ( false );
  }
#endif
  SHOW_TENSOR_OP(self, mat1, mat2, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addmm.out", addmm_out_tpu );
}

Tensor & mm_out_tpu ( const Tensor & self, const Tensor & mat2, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = mm ( self.cpu().to(torch::kFloat32), mat2.cpu().to(torch::kFloat32) ).to(out.dtype());
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  auto out_cpu = mm ( self.cpu().to(torch::kFloat32), mat2.cpu().to(torch::kFloat32) );
  out = out_cpu.to(out.device()).to(out.dtype());
#else
  auto self_ = self.is_contiguous() == false && is_transposed ( self ) == false ? self.contiguous() : self;
  auto mat2_ = mat2.is_contiguous() == false && is_transposed ( mat2 ) == false ? mat2.contiguous() : mat2;

  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnMatmulAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, mat2_),
    tpudnnUndefinedTensor(),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
    TIMING_END( tpu::MM );
#endif
  SHOW_TENSOR_OP(self, mat2, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mm.out", mm_out_tpu );
}

Tensor & bmm_out_tpu ( const Tensor & self, const Tensor & mat2, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = bmm ( self.cpu().to(torch::kFloat32), mat2.cpu().to(torch::kFloat32) ).to(out.dtype());
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto self_ = self.is_contiguous() == false && is_transposed ( self ) == false ? self.contiguous() : self;
  auto mat2_ = mat2.is_contiguous() == false && is_transposed ( mat2 ) == false ? mat2.contiguous() : mat2;

  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnMatmulAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, mat2_),
    tpudnnUndefinedTensor(),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );

  TIMING_END( tpu::BMM );
#endif
  SHOW_TENSOR_OP(self, mat2, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "bmm.out", bmm_out_tpu );
}


Tensor & baddbmm_out_tpu(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar & beta, const Scalar & alpha, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( batch1 );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( batch2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  CPU_IMPL_WARNING();
  auto out_cpu = baddbmm ( self.to(torch::kFloat).cpu(), batch1.to(torch::kFloat).cpu(), batch2.to(torch::kFloat).cpu(), beta, alpha );
  out = out_cpu.to(out.device()).to(out.dtype());
#else
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  auto batch1_ = batch1.is_contiguous() == false && is_transposed ( batch1 ) == false ? batch1.contiguous() : batch1;
  auto batch2_ = batch2.is_contiguous() == false && is_transposed ( batch2 ) == false ? batch2.contiguous() : batch2;
  if (beta.toDouble() != 0)
    out = beta * self_ + alpha * bmm(batch1_, batch2_);
  else
    out = alpha * bmm(batch1_, batch2_);

#endif
  SHOW_TENSOR_OP(self, batch1, batch2, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "baddbmm.out", baddbmm_out_tpu );
}

Tensor & addbmm_out_tpu(const Tensor & self, const Tensor & batch1, const Tensor & batch2,
    const Scalar & beta, const Scalar & alpha, Tensor & out) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  auto batch1_ = batch1.is_contiguous() == false && is_transposed ( batch1 ) == false ? batch1.contiguous() : batch1;
  auto batch2_ = batch2.is_contiguous() == false && is_transposed ( batch2 ) == false ? batch2.contiguous() : batch2;
  if (beta.toDouble() != 0)
    out = beta * self_ + alpha * bmm(batch1_, batch2_).sum({0});
  else
    out = alpha * bmm(batch1_, batch2_).sum({0});
  return out;
}
Tensor addbmm_tpu(const Tensor & self, const Tensor & batch1, const Tensor & batch2,
    const Scalar & beta, const Scalar & alpha) {
  auto out = empty({batch1.size(1), batch2.size(2)}, batch1.options());
  return addbmm_out_tpu(self, batch1, batch2, beta, alpha, out);
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addbmm.out", addbmm_out_tpu );
  m.impl ( "addbmm",     addbmm_tpu );
}

// Tensor & addmv__tpu(Tensor & self, const Tensor & mat, const Tensor & vec,
//     const Scalar & beta, const Scalar & alpha) {}

Tensor & addmv_out_tpu(const Tensor & self, const Tensor & mat, const Tensor & vec,
    const Scalar & beta, const Scalar & alpha, Tensor & out) {
  std::vector<int64_t> mat2_sizes_vec;
  for (int i = 0; i < vec.dim(); i++) { mat2_sizes_vec.push_back(vec.size(i)); } mat2_sizes_vec.push_back(1);
  IntArrayRef mat2_sizes(mat2_sizes_vec);
  auto vec_mat = vec.view(mat2_sizes);
  out = beta * self + mm(mat, vec_mat).view(out.sizes()) * alpha;
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addmv.out", addmv_out_tpu );
  // m.impl ( "addmv_", addmv__tpu );
}

Tensor & cholesky_out_tpu(const Tensor & self, bool upper, Tensor & out) {
  if (upper){ mm_out_tpu(self.transpose(1, 0), self, out); }
  else      { mm_out_tpu(self, self.transpose(1, 0), out); }
  return out;
}
Tensor cholesky_tpu(const Tensor & self, bool upper) {
  Tensor out;
  if (upper) { out = empty({self.size(1), self.size(1)}, self.options());}
  else       { out = empty({self.size(0), self.size(0)}, self.options());}
  out = cholesky_out_tpu(self, upper, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cholesky.out", cholesky_out_tpu );
  m.impl ( "cholesky",     cholesky_tpu );
}

std::tuple<Tensor &,Tensor &> linalg_cholesky_ex_out_tpu(const Tensor & self, bool upper,
    bool check_errors, Tensor & L, Tensor & info) {
  CPU_IMPL_WARNING();
  auto outputs_cpu = linalg_cholesky_ex(self.cpu(), upper, check_errors);
  L    = std::get<0> (outputs_cpu).to(L.device());
  info = std::get<1> (outputs_cpu).to(info.device()); 
  return {L, info};
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "linalg_cholesky_ex.L", linalg_cholesky_ex_out_tpu );
}

std::tuple<Tensor &,Tensor &> linalg_inv_ex_out_tpu(const Tensor & A, bool check_errors, Tensor & inverse, Tensor & info) {
  CPU_IMPL_WARNING();
  auto outputs_cpu = linalg_inv_ex(A.cpu(), check_errors);
  inverse    = std::get<0> (outputs_cpu).to(inverse.device());
  info       = std::get<1> (outputs_cpu).to(info.device()); 
  return {inverse, info};
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "linalg_inv_ex.inverse", linalg_inv_ex_out_tpu );
}

Tensor & dot_out_tpu(const Tensor & self, const Tensor & tensor, Tensor & out) {
  TORCH_CHECK(self.dim() == 1, "self should be a vector");
  TORCH_CHECK(tensor.dim() == 1, "other should be a vector");
  auto out_1D = out.view({1, 1});
  mm_out_tpu(self.view({1, self.size(0)}), tensor.view({tensor.size(0), 1}), out_1D);
  return out;
}
Tensor dot_tpu(const Tensor & self, const Tensor & tensor) {
  auto out = empty({}, self.options());
  dot_out_tpu(self, tensor, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "dot.out", dot_out_tpu );
  m.impl ( "dot",     dot_tpu );
}

std::tuple<Tensor &,Tensor &,Tensor &> _linalg_det_out_tpu(const Tensor & A, Tensor & result, Tensor & LU,Tensor & pivots) {
  CPU_IMPL_WARNING();
  auto outs_cpu = _linalg_det(A.cpu());
  result = std::get<0>(outs_cpu).to(result.device());
  LU     = std::get<1>(outs_cpu).to(LU.device());
  pivots = std::get<2>(outs_cpu).to(pivots.device());
  return {result, LU, pivots};
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_linalg_det.result", _linalg_det_out_tpu );
}

std::tuple<Tensor &,Tensor &,Tensor &,Tensor &> _linalg_slogdet_out_tpu(const Tensor & A, Tensor & sign,
    Tensor & logabsdet, Tensor & LU, Tensor & pivots) {
  CPU_IMPL_WARNING();
  auto outs_cpu = _linalg_slogdet(A.cpu());
  sign      = std::get<0>(outs_cpu).to(sign.device());
  logabsdet = std::get<1>(outs_cpu).to(logabsdet.device());
  LU        = std::get<2>(outs_cpu).to(LU.device());
  pivots    = std::get<3>(outs_cpu).to(pivots.device());
  return {sign, logabsdet, LU, pivots};
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_linalg_slogdet.sign", _linalg_slogdet_out_tpu );
}
} // namespace at
