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
Tensor & signbit_out_tpu ( const Tensor & self, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = signbit( self.cpu());
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#endif
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnSignbit(
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                    
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::SIGNBIT, timer.ElapsedUS() );
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "signbit.out", signbit_out_tpu );
  
}
} // namespace at
