#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

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
  bm_status_t status = sgdnn_const_fill_cudnn(
    tpu::TPUGetDeviceHandle(),
    tpu::TPUGenerateTensorDesc( self ),
    ADDR_IN_DEVICE( self ),
    value.data_ptr());
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
  fill__Scalar_tpu(self, 0);
#endif
  return self;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "zero_", zero__tpu );
}
} // namespace at
