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
Tensor & fill__Scalar_tpu ( Tensor & self, const Scalar & value )
{
  CHECK_TENSOR_IN_DEVICE ( self );
#if 0
  auto self_cpu = TENSOR_TO_CPU ( self );
  self_cpu.fill_ ( value );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto self_ = self.dim() == 0 ? self.unsqueeze ( 0 ) : self;
  int64_t value_;
  if ( self.dtype() == caffe2::TypeMeta::Make<float>() ) {
    * ( float * ) ( &value_ ) = value.toFloat();
  } else if ( self.dtype() == caffe2::TypeMeta::Make<at::Half>() ) {
    * ( at::Half * ) ( &value_ ) = value.toHalf();
  } else {
    TORCH_CHECK ( false );
  }
  bm_status_t status = sgdnnFill (
                       tpu::TPUGetDeviceHandle(),
                       &value_,
                       tpu::TPUGenerateSgdnnTensor ( self_ ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::CONST_FILL, timer.ElapsedUS() );
#endif
#endif
  return self;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "fill_.Scalar", fill__Scalar_tpu );
}

Tensor & zero__tpu ( Tensor & self )
{
  CHECK_TENSOR_IN_DEVICE ( self );
#if 0
  char * buffer = new char [self.nbytes()];
  memset ( buffer, 0x0, self.nbytes() );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), buffer, self.nbytes() );
  delete [] buffer;
#else
  fill__Scalar_tpu ( self, 0 );
#endif
  return self;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "zero_", zero__tpu );
}

Tensor & masked_fill_Scalar_tpu ( Tensor & self, const Tensor & mask, const Scalar & value)
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mask );
#if 1
  LOG( WARNING ) << "masked_fill_.Scalar use cpu impl";
  auto self_cpu = self.cpu();
  self_cpu.masked_fill_(mask.cpu(), value);
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
#else
  // TODO
#endif
  return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "masked_fill_.Scalar", masked_fill_Scalar_tpu );
}

} // namespace at
