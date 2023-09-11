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



Tensor & bitwise_not_out_tpu ( const Tensor & self, Tensor & out )
{
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  
#else
  if ( self.dim() == 0)
  {
    auto out_cpu = bitwise_not ( self.cpu());
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ))
  {
    
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnBitwiseNot (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::BITWISE_NOT, timer.ElapsedUS() );
#endif

  }
#endif
  return out;
}

Tensor bitwise_not_tpu(const Tensor &self){
  auto out = empty_like(self);
  return bitwise_not_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "bitwise_not", bitwise_not_tpu);
  m.impl ( "bitwise_not.out", bitwise_not_out_tpu );
}
} // namespace at
