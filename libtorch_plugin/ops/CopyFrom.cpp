#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <c10/util/Logging.h>

#include <iostream>

namespace at
{
Tensor _copy_from_tpu ( const Tensor & self, const Tensor & dst,
                        bool non_blocking )
{
  auto output = dst;
  if ( self.is_contiguous() == false || dst.is_contiguous() == false )
  {
    LOG ( FATAL ) << "TPU only supports contiguous memory copy for now";
  }
  if ( self.dtype() == dst.dtype() )
  {
    if ( self.device().type() == DeviceType::CPU &&
         dst.device().type() == DeviceType::PrivateUse1 )
    {
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), self.data_ptr(),
                                 dst.nbytes() );
    }
    else if ( self.device().type() == DeviceType::PrivateUse1 &&
              dst.device().type() == DeviceType::CPU )
    {
      tpu::TPUCopyDeviceToHost ( dst.data_ptr(), self.data_ptr(),
                                 dst.nbytes() );
    }
    else if ( self.device().type() == DeviceType::PrivateUse1 &&
              dst.device().type() == DeviceType::PrivateUse1 )
    {
      tpu::TPUCopyDeviceToDevice ( dst.data_ptr(), self.data_ptr(),
                                   dst.nbytes() );
    }
    else
    {
      LOG ( FATAL ) << "Unsupported copy from device " << self.device()
                    << " to device " << dst.device();
    }
  }
  else
  {
    if ( self.device().type() == DeviceType::CPU &&
         dst.device().type() == DeviceType::PrivateUse1 )
    {
      // SELF CPU -> SELF TPU -> DST TPU
      auto SelfTPU = self.to ( dst.device() );
      _copy_from_tpu ( SelfTPU, dst, non_blocking );
    }
    else if ( self.device().type() == DeviceType::PrivateUse1 &&
              dst.device().type() == DeviceType::CPU )
    {
      // SELF TPU -> DST TPU -> DST CPU
      auto DstTPU = self.to ( dst.dtype() );
      _copy_from_tpu ( DstTPU, dst, non_blocking );
    }
    else if ( self.device().type() == DeviceType::PrivateUse1 &&
              dst.device().type() == DeviceType::PrivateUse1 )
    {
      LOG ( FATAL ) << "Cast from " << self.dtype()
                    << " to " << dst.dtype() << " is not implemented";
    }
    else
    {
      LOG ( FATAL ) << "Unsupported copy from device " << self.device()
                    << " to device " << dst.device();
    }
  }
  return output;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_copy_from", _copy_from_tpu );
}
} // namespace at
