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
    if ( tpu::TPUIsSameShape ( self, other ) )
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnAdd (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           alpha.toDouble(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD, timer.ElapsedUS() );
#endif
    }
    else
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnAddBcast (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           alpha.toDouble(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BCAST_ADD, timer.ElapsedUS() );
#endif
    }
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
      bm_status_t status = sgdnnAddC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           alpha.toDouble() * ( *scalar.data_ptr<float>() ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD_C, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( alpha.toDouble() == 1.0 );
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
      bm_status_t status = sgdnnAddC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD_C, timer.ElapsedUS() );
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
  if ( self.dim() == 0 && other.dim() == 0 )
  {
    auto out_cpu = sub ( self.cpu(), other.cpu(), alpha );
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) )
  {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnSub (
                         tpu::TPUGetDeviceHandle(),
                         tpu:: TPUGenerateSgdnnTensor ( self ),
                         tpu:: TPUGenerateSgdnnTensor ( other ),
                         alpha.toDouble(),
                         tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::SUB, timer.ElapsedUS() );
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
      bm_status_t status = sgdnnAddC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           - alpha.toDouble() * ( *scalar.data_ptr<float>() ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::ADD_C, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( alpha.toDouble() == 1.0 );
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
      bm_status_t status = sgdnnCSub (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           alpha.toDouble() * ( *scalar.data_ptr<float>() ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::C_SUB, timer.ElapsedUS() );
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
  if ( self.dim() == 0 && other.dim() == 0 )
  {
    auto out_cpu = mul ( self.cpu(), other.cpu() );
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) )
  {
    if ( other.dim() == 0 )
    {
      Tensor scalar = other.cpu().to ( torch::kFloat );
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL_C, timer.ElapsedUS() );
#endif
    }
    else
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMul (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL, timer.ElapsedUS() );
#endif
    }
  }
  else if ( ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) ) )
  {
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      Tensor scalar = other.to ( torch::kFloat );
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL_C, timer.ElapsedUS() );
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
      bm_status_t status = sgdnnMulC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL_C, timer.ElapsedUS() );
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
  if ( self.dim() == 0 && other.dim() == 0 )
  {
    auto out_cpu = div ( self.cpu(), other.cpu() );
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self_ ) && IS_TPU_TENSOR ( other ) )
  {
    if ( other.dim() == 0 )
    {
      /* RECIPROCAL */
      Tensor scalar = ( 1.0 / other.cpu().to ( torch::kFloat ) ).to ( torch::kFloat );
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self_ ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL_C, timer.ElapsedUS() );
#endif
    }
    else
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnDiv (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self_ ),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::DIV, timer.ElapsedUS() );
#endif
    }
  }
  else if ( ( IS_TPU_TENSOR ( self_ ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self_ ) && IS_TPU_TENSOR ( other ) ) )
  {
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      /* RECIPROCAL */
      Tensor scalar = ( 1.0 / other.to ( torch::kFloat ) ).to ( torch::kFloat );
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self_ ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::MUL_C, timer.ElapsedUS() );
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
      bm_status_t status = sgdnnCDiv (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( other ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::C_DIV, timer.ElapsedUS() );
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

Tensor &bitwise_xor_out_tpu ( const Tensor & self, const Tensor & other, Tensor & out ) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = bitwise_xor(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnBitwiseXorC(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( other ),
                                          self.item().toInt(),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_XOR_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnBitwiseXorC(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( self ),
                                          other.item().toInt(),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_XOR_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
    if(tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseXor(tpu::TPUGetDeviceHandle(),
                                           tpu:: TPUGenerateSgdnnTensor ( self ),
                                           tpu:: TPUGenerateSgdnnTensor ( other ),
                                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_XOR, timer.ElapsedUS() );
#endif

    }
    else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseXorBcast(tpu::TPUGetDeviceHandle(),
                                                tpu:: TPUGenerateSgdnnTensor ( self ),
                                                tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                                tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_XOR_BCAST, timer.ElapsedUS() );
#endif

    }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m ) {
  m.impl ( "bitwise_xor.Tensor_out", bitwise_xor_out_tpu );
}

Tensor &bitwise_and_out_tpu ( const Tensor & self, const Tensor & other, Tensor & out ) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = bitwise_and(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnBitwiseAndC(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( other ),
                                          self.item().toInt(),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_AND_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnBitwiseAndC(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( self ),
                                          other.item().toInt(),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_AND_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
    if(tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseAnd(tpu::TPUGetDeviceHandle(),
                                           tpu:: TPUGenerateSgdnnTensor ( self ),
                                           tpu:: TPUGenerateSgdnnTensor ( other ),
                                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_AND, timer.ElapsedUS() );
#endif

    }
    else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseAndBcast(tpu::TPUGetDeviceHandle(),
                                                tpu:: TPUGenerateSgdnnTensor ( self ),
                                                tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                                tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_AND_BCAST, timer.ElapsedUS() );
#endif

    }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m ) {
  m.impl ( "bitwise_and.Tensor_out", bitwise_and_out_tpu );
}

Tensor &bitwise_or_out_tpu ( const Tensor & self, const Tensor & other, Tensor & out ) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = bitwise_or(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnBitwiseOrC(tpu::TPUGetDeviceHandle(),
                                         tpu:: TPUGenerateSgdnnTensor ( other ),
                                         self.item().toInt(),
                                         tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_OR_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnBitwiseOrC(tpu::TPUGetDeviceHandle(),
                                         tpu:: TPUGenerateSgdnnTensor ( self ),
                                         other.item().toInt(),
                                         tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_OR_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
    if(tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseOr(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( self ),
                                          tpu:: TPUGenerateSgdnnTensor ( other ),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_OR, timer.ElapsedUS() );
#endif

    }
    else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseOrBcast(tpu::TPUGetDeviceHandle(),
                                               tpu:: TPUGenerateSgdnnTensor ( self ),
                                               tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                               tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_OR_BCAST, timer.ElapsedUS() );
#endif

    }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m ) {
  m.impl ( "bitwise_or.Tensor_out", bitwise_or_out_tpu );
}

Tensor & equal_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = eq(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnEqualC(tpu::TPUGetDeviceHandle(),
                                       tpu:: TPUGenerateSgdnnTensor ( other ),
                                       self.item().toFloat(),
                                       tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnEqualC(tpu::TPUGetDeviceHandle(),
                                       tpu:: TPUGenerateSgdnnTensor ( self ),
                                       other.item().toFloat(),
                                       tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
      if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnEqual(tpu::TPUGetDeviceHandle(),
                                      tpu:: TPUGenerateSgdnnTensor ( self ),
                                      tpu:: TPUGenerateSgdnnTensor ( other ),
                                      tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::EQUAL, timer.ElapsedUS() );
#endif

      }
      else {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnEqualBcast(tpu::TPUGetDeviceHandle(),
                                                 tpu:: TPUGenerateSgdnnTensor ( self ),
                                                 tpu:: TPUGenerateSgdnnTensor ( other ),
                                                 tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime ( tpu::EQUAL_BCAST, timer.ElapsedUS() );
#endif

      }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "eq.Tensor_out", equal_out_tpu );
}

Tensor & greater_or_equal_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = ge(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnGeC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                    self.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::GREATER_OR_EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnGeC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                    other.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::GREATER_OR_EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
      if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnGe(tpu::TPUGetDeviceHandle(),
                                   tpu:: TPUGenerateSgdnnTensor ( self ),
                                   tpu:: TPUGenerateSgdnnTensor ( other ),
                                   tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::GREATER_OR_EQUAL, timer.ElapsedUS() );
#endif

      }
      else {
#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnGeBcast(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime ( tpu::GREATER_OR_EQUAL_BCAST, timer.ElapsedUS() );
#endif

      }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "ge.Tensor_out", greater_or_equal_out_tpu );
}

Tensor & greater_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = gt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnGtC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                    self.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::GREATER_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnGtC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                    other.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::GREATER_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
      if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnGt(tpu::TPUGetDeviceHandle(),
                                   tpu:: TPUGenerateSgdnnTensor ( self ),
                                   tpu:: TPUGenerateSgdnnTensor ( other ),
                                   tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::GREATER, timer.ElapsedUS() );
#endif

      }
      else {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnGtBcast(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime ( tpu::GREATER_BCAST, timer.ElapsedUS() );
#endif

      }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "gt.Tensor_out", greater_out_tpu );
}

Tensor & less_than_or_equal_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = le(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnLeC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                    self.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_OR_EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnLeC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                    other.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_OR_EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
      if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnLe(tpu::TPUGetDeviceHandle(),
                                   tpu:: TPUGenerateSgdnnTensor ( self ),
                                   tpu:: TPUGenerateSgdnnTensor ( other ),
                                   tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_OR_EQUAL, timer.ElapsedUS() );
#endif

      }
      else {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnLeBcast(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_OR_EQUAL_BCAST, timer.ElapsedUS() );
#endif

      }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "le.Tensor_out", less_than_or_equal_out_tpu );
}

Tensor & less_than_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = lt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnLtC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                    self.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnLtC(tpu::TPUGetDeviceHandle(),
                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                    other.item().toFloat(),
                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
      if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnLt(tpu::TPUGetDeviceHandle(),
                                   tpu:: TPUGenerateSgdnnTensor ( self ),
                                   tpu:: TPUGenerateSgdnnTensor ( other ),
                                   tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN, timer.ElapsedUS() );
#endif

      }
      else {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnLtBcast(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_BCAST, timer.ElapsedUS() );
#endif

      }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "lt.Tensor_out", less_than_out_tpu );
}

Tensor & not_equal_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = ne(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnNotEqualC(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( other ),
                                          self.item().toFloat(),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::NOT_EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnNotEqualC(tpu::TPUGetDeviceHandle(),
                                          tpu:: TPUGenerateSgdnnTensor ( self ),
                                          other.item().toFloat(),
                                          tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::NOT_EQUAL_C, timer.ElapsedUS() );
#endif

  }
  else if(self.dim() == other.dim()) {
      if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnNotEqual(tpu::TPUGetDeviceHandle(),
                                         tpu:: TPUGenerateSgdnnTensor ( self ),
                                         tpu:: TPUGenerateSgdnnTensor ( other ),
                                         tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::NOT_EQUAL, timer.ElapsedUS() );
#endif

      }
      else {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnNotEqualBcast(tpu::TPUGetDeviceHandle(),
                                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                                    tpu:: TPUGenerateSgdnnTensor ( other ),                                              
                                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime ( tpu::NOT_EQUAL_BCAST, timer.ElapsedUS() );
#endif

      }
  }
  else {
    TORCH_CHECK ( false, "unsupported dims" );
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "ne.Tensor_out", not_equal_out_tpu );
}

} // namespace at
