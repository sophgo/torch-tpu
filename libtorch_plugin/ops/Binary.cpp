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
Tensor & add_out_tpu ( const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = add ( self.cpu(), other.cpu(), alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( self.dim() == 0 && other.dim() == 0 )
  {
    auto out_cpu = add ( self.cpu(), other.cpu(), alpha );
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) )
  {
    if ( alpha.toDouble() == 1.0 )
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD, timer.ElapsedUS() );
#endif
    }
    else if ( alpha.toDouble() == -1.0 )
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_SUB );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::SUB, timer.ElapsedUS() );
#endif
    }
    else
    {
      add_out_tpu ( self, other * alpha, 1.0, out );
    }
  }
  else if ( ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) ) )
  {
    TORCH_CHECK ( alpha.toDouble() == 1.0 );
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      Tensor scalar;
      if ( other.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = other.to ( torch::kFloat );
      }
      else
      {
        scalar = other;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( self.dim() == 0, "SELF must be a scalar" );
      Tensor scalar;
      if ( self.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = self.to ( torch::kFloat );
      }
      else
      {
        scalar = self;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD, timer.ElapsedUS() );
#endif
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "add.out", add_out_tpu );
}

Tensor & sub_out_tpu ( const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = sub ( self.cpu(), other.cpu(), alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) )
  {
    if ( alpha.toDouble() == 1.0 )
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_SUB );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::SUB, timer.ElapsedUS() );
#endif
    }
    else if ( alpha.toDouble() == -1.0 )
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD, timer.ElapsedUS() );
#endif
    }
    else
    {
      sub_out_tpu ( self, other * alpha, 1.0, out );
    }
  }
  else if ( ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) ) )
  {
    TORCH_CHECK ( alpha.toDouble() == 1.0 );
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      Tensor scalar;
      if ( other.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = other.to ( torch::kFloat );
      }
      else
      {
        scalar = other;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_SUB );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::SUB, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( self.dim() == 0, "SELF must be a scalar" );
      Tensor scalar;
      if ( self.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = self.to ( torch::kFloat );
      }
      else
      {
        scalar = self;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_SUB );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::SUB, timer.ElapsedUS() );
#endif
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "sub.out", sub_out_tpu );
}

Tensor & mul_out_tpu ( const Tensor & self, const Tensor & other, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = mul ( self.cpu(), other.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) )
  {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnn_binary (
                         tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateTensorDesc ( self ),
                         ADDR_IN_DEVICE ( self ),
                         tpu::TPUGenerateTensorDesc ( other ),
                         ADDR_IN_DEVICE ( other ),
                         tpu::TPUGenerateTensorDesc ( out ),
                         ADDR_IN_DEVICE ( out ),
                         OP_BINARY_MUL );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::MUL, timer.ElapsedUS() );
#endif
  }
  else if ( ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) ) )
  {
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      Tensor scalar;
      if ( other.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = other.to ( torch::kFloat );
      }
      else
      {
        scalar = other;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self ),
                           ADDR_IN_DEVICE ( self ),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_MUL );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( self.dim() == 0, "SELF must be a scalar" );
      Tensor scalar;
      if ( self.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = self.to ( torch::kFloat );
      }
      else
      {
        scalar = self;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_MUL );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL, timer.ElapsedUS() );
#endif
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mul.out", mul_out_tpu );
}

Tensor & div_out_tpu ( const Tensor & self, const Tensor & other, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE ( out );
  auto self_ = self.contiguous();
#if 0
  auto out_cpu = div ( self.cpu(), other.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( IS_TPU_TENSOR ( self_ ) && IS_TPU_TENSOR ( other ) )
  {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnn_binary (
                         tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateTensorDesc ( self_ ),
                         ADDR_IN_DEVICE ( self_ ),
                         tpu::TPUGenerateTensorDesc ( other ),
                         ADDR_IN_DEVICE ( other ),
                         tpu::TPUGenerateTensorDesc ( out ),
                         ADDR_IN_DEVICE ( out ),
                         OP_BINARY_DIV );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::DIV, timer.ElapsedUS() );
#endif
  }
  else if ( ( IS_TPU_TENSOR ( self_ ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self_ ) && IS_TPU_TENSOR ( other ) ) )
  {
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      Tensor scalar;
      if ( other.dtype() == caffe2::TypeMeta::Make<double>() ||
           other.dtype() == caffe2::TypeMeta::Make<long>() )
      {
        scalar = other.to ( torch::kFloat );
      }
      else
      {
        scalar = other;
      }
      /* RECIPROCAL */
      scalar = 1.0 / scalar;
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( self_ ),
                           ADDR_IN_DEVICE ( self_ ),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_MUL );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( self_.dim() == 0, "SELF must be a scalar" );
      Tensor scalar;
      if ( self_.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = self_.to ( torch::kFloat );
      }
      else
      {
        scalar = self_;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnn_binary (
                           tpu::TPUGetDeviceHandle(),
                           tpu::TPUGenerateTensorDesc ( scalar ),
                           scalar.data_ptr(),
                           tpu::TPUGenerateTensorDesc ( other ),
                           ADDR_IN_DEVICE ( other ),
                           tpu::TPUGenerateTensorDesc ( out ),
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_DIV );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::DIV, timer.ElapsedUS() );
#endif
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "div.out", div_out_tpu );
}
} // namespace at
