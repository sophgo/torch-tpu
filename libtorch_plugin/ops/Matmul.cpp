#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

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
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "Addmm " << count << std::endl;
  ++count;
#endif
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
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    auto status = sgdnnMatmul (
                  tpu::TPUGetDeviceHandle(),
                  tpu::TPUGenerateSgdnnTensor ( mat1_ ),
                  tpu::TPUGenerateSgdnnTensor ( mat2_ ),
                  tpu::TPUGenerateSgdnnTensor ( self ),
                  tpu::TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::MM, timer.ElapsedUS() );
#endif
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else
  {
    TORCH_CHECK ( false );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addmm.out", addmm_out_tpu );
}

Tensor & mm_out_tpu ( const Tensor & self, const Tensor & mat2, Tensor & out )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "mm " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = mm ( self.cpu(), mat2.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto self_ = self.is_contiguous() == false && is_transposed ( self ) == false ? self.contiguous() : self;
  auto mat2_ = mat2.is_contiguous() == false && is_transposed ( mat2 ) == false ? mat2.contiguous() : mat2;
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto status = sgdnnMatmul (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor ( self_ ),
                tpu::TPUGenerateSgdnnTensor ( mat2_ ),
                sgdnnUndefinedTensor(),
                tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::MM, timer.ElapsedUS() );
#endif
  TORCH_CHECK ( status == BM_SUCCESS );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mm.out", mm_out_tpu );
}

Tensor & bmm_out_tpu ( const Tensor & self, const Tensor & mat2, Tensor & out )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "bmm " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = bmm ( self.cpu(), mat2.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto self_ = self.is_contiguous() == false && is_transposed ( self ) == false ? self.contiguous() : self;
  auto mat2_ = mat2.is_contiguous() == false && is_transposed ( mat2 ) == false ? mat2.contiguous() : mat2;
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto status = sgdnnBatchMatmul (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor ( self_ ),
                tpu::TPUGenerateSgdnnTensor ( mat2_ ),
                tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BMM, timer.ElapsedUS() );
#endif
#endif
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
#if 1
  LOG( WARNING ) << "baddbmm use cpu impl";
  auto out_cpu = baddbmm ( self.to(torch::kFloat).cpu(), batch1.to(torch::kFloat).cpu(), batch2.to(torch::kFloat).cpu(), beta, alpha );
  out = out_cpu.to(out.device()).to(out.dtype());
#else
//TODO
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "baddbmm.out", baddbmm_out_tpu );
}
} // namespace at
