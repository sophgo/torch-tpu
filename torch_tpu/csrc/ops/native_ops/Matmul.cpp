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


Tensor & baddbmm_out_tpu(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out )
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

  // TIMING_START;
  //
  // // TODO: imple this op, current has bugs
  // auto status = sgdnnBaddbmm(
  //               tpu::TPUGetDeviceResource(),
  //               tpu::TPUGenerateSgdnnTensor ( self_ ),
  //               tpu::TPUGenerateSgdnnTensor ( batch1_ ),
  //               tpu::TPUGenerateSgdnnTensor ( batch2_ ),
  //               tpu::TPUGenerateSgdnnTensor ( out ),
  //               alpha.toDouble(),
  //               beta.toDouble() );
  //   // TIMING_END( tpu::BADDBMM );
#endif
  SHOW_TENSOR_OP(self, batch1, batch2, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "baddbmm.out", baddbmm_out_tpu );
}
} // namespace at
