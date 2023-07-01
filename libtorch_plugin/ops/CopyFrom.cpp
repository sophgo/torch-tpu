#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <c10/util/Logging.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <iostream>

#include "common/config.h"

namespace at
{
Tensor _copy_from_tpu ( const Tensor & self, const Tensor & dst, bool non_blocking )
{
  TORCH_CHECK ( non_blocking == false );
  if ( self.dtype() == dst.dtype() )
  {
    TORCH_CHECK ( self.nbytes() == dst.nbytes(), "SELF and dst number bytes must be the same" );
    if ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
      if ( dst.is_contiguous() ) {
        tpu::TPUCopyHostToDevice ( dst.data_ptr(), self.contiguous().data_ptr(), dst.nbytes() );
      } else {
        dst.copy_ ( self.contiguous().to ( dst.device() ), non_blocking );
      }
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( dst ) )
    {
      if ( dst.is_contiguous() ) {
        tpu::TPUCopyDeviceToHost ( dst.data_ptr(), self.contiguous().data_ptr(), dst.nbytes() );
      } else {
        dst.copy_ ( self.contiguous().to ( dst.device() ), non_blocking );
      }
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
      if ( self.is_contiguous() && dst.is_contiguous() ) {
        tpu::TPUCopyDeviceToDevice ( dst.data_ptr(), self.data_ptr(), dst.nbytes() );
      } else {
#ifdef TPU_OP_TIMING
        auto timer = tpu::Timer().Start();
#endif
        bm_status_t status = sgdnnStridedCopy (
                             tpu::TPUGetDeviceHandle(),
                             tpu::TPUGenerateSgdnnTensor ( self ),
                             tpu::TPUGenerateSgdnnTensor ( dst ) );
        TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
        tpu::OpTimer::Instance().AddTime ( tpu::STRIDED_COPY, timer.ElapsedUS() );
#endif
      }
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported copy from device ", self.device(), " to device ", dst.device() );
    }
  }
  else
  {
    if ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
      dst.copy_ ( self.to ( dst.device() ), non_blocking );
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( dst ) )
    {
      dst.copy_ ( self.to ( dst.dtype() ), non_blocking );
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
#if 0
      auto dst_cpu = self.cpu().to ( dst.dtype() );
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), dst_cpu.contiguous().data_ptr(), dst.nbytes() );
#else
      auto self_ = self.contiguous();
      if ( dst.is_contiguous() ) {
#ifdef TPU_OP_TIMING
        auto timer = tpu::Timer().Start();
#endif
        auto status = sgdnnConvert (
                      tpu::TPUGetDeviceHandle(),
                      tpu::TPUGenerateSgdnnTensor ( self_ ),
                      tpu::TPUGenerateSgdnnTensor ( dst ) );
        TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
        tpu::OpTimer::Instance().AddTime ( tpu::DTYPE_CONVERT, timer.ElapsedUS() );
#endif
      } else {
        dst.copy_ ( self_.to ( dst.dtype() ), non_blocking );
      }
#endif
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported copy from device ", self.device(), " to device ", dst.device() );
    }
  }
  return dst;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_copy_from", _copy_from_tpu );
}
} // namespace at
