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
Tensor & asin_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = asin ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnAsin (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ASIN, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor asin_tpu(const Tensor &self){
  auto out = empty_like(self);
  return asin_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "asin", asin_tpu);
  m.impl ( "asin.out", asin_out_tpu );
}

Tensor & acos_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = acos ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnAcos (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ACOS, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor acos_tpu(const Tensor &self){
  auto out = empty_like(self);
  return acos_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "acos", acos_tpu);
  m.impl ( "acos.out", acos_out_tpu );
}

Tensor & atan_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = atan ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnAtan (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ATAN, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor atan_tpu(const Tensor &self){
  auto out = empty_like(self);
  return atan_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "atan", atan_tpu);
  m.impl ( "atan.out", atan_out_tpu );
}

Tensor & ceil_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = ceil ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnCeil (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::CEIL, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor ceil_tpu(const Tensor &self){
  auto out = empty_like(self);
  return ceil_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "ceil", ceil_tpu);
  m.impl ( "ceil.out", ceil_out_tpu );
}


Tensor & floor_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = floor ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnFloor (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::FLOOR, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor floor_tpu(const Tensor &self){
  auto out = empty_like(self);
  return floor_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "floor", floor_tpu);
  m.impl ( "floor.out", floor_out_tpu );
}

Tensor & round_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = round ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnRound (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ROUND, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor round_tpu(const Tensor &self){
  auto out = empty_like(self);
  return round_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "round", round_tpu);
  m.impl ( "round.out", round_out_tpu );
}


Tensor & exp2_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = exp2 ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnExp2 (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::EXP2, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor exp2_tpu(const Tensor &self){
  auto out = empty_like(self);
  return exp2_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "exp2", exp2_tpu);
  m.impl ( "exp2.out", exp2_out_tpu );
}

Tensor isfinite_tpu ( const Tensor & self )
{
  auto out = empty_like(self,kBool);

  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = isfinite ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnIsfinite (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ISFINITE, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "isfinite", isfinite_tpu );
}

} // namespace at
