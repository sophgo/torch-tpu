#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <c10/util/Logging.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <iostream>

#define TPU_OP_TIMING

namespace at
{
Tensor _copy_from_tpu ( const Tensor & self, const Tensor & dst, bool non_blocking )
{
  TORCH_CHECK ( non_blocking == false );
  if ( self.is_contiguous() == false || dst.is_contiguous() == false )
  {
    TORCH_CHECK ( false, "TPU only supports contiguous memory copy for now" );
  }
  if ( self.dtype() == dst.dtype() )
  {
    TORCH_CHECK ( self.nbytes() == dst.nbytes(), "SELF and dst number bytes must be the same" );
    if ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), self.data_ptr(), dst.nbytes() );
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( dst ) )
    {
      tpu::TPUCopyDeviceToHost ( dst.data_ptr(), self.data_ptr(), dst.nbytes() );
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
      tpu::TPUCopyDeviceToDevice ( dst.data_ptr(), self.data_ptr(), dst.nbytes() );
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
      _copy_from_tpu ( self.to ( dst.device() ), dst, non_blocking );
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( dst ) )
    {
      _copy_from_tpu ( self.to ( dst.dtype() ), dst, non_blocking );
    }
    else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( dst ) )
    {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
#if 1
      auto dst_cpu = self.cpu().to ( dst.dtype() );
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), dst_cpu.contiguous().data_ptr(), dst.nbytes() );
#else
      auto status = sgdnn_dtype_convert (
                    tpu::TPUGetDeviceHandle(),
                    tpu::TPUGenerateTensorDesc ( self ),
                    ADDR_IN_DEVICE ( self ),
                    tpu::TPUGenerateTensorDesc ( dst ),
                    ADDR_IN_DEVICE ( dst ),
                    SG_ROUND_EVEN );
      TORCH_CHECK ( status == BM_SUCCESS );
#endif
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::DTYPE_CONVERT, timer.ElapsedUS() );
#endif
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported copy from device ", self.device(), " to device ", dst.device() );
    }
  }
  return dst;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_copy_from", _copy_from_tpu );
}
} // namespace at
