#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <c10/util/Logging.h>

#include <iostream>

namespace at
{
Tensor _copy_from_tpu ( const Tensor & src, const Tensor & dst, bool non_blocking )
{
  TORCH_CHECK ( non_blocking == false );
  if ( src.is_contiguous() == false || dst.is_contiguous() == false )
  {
    LOG ( FATAL ) << "TPU only supports contiguous memory copy for now";
  }
  if ( src.dtype() == dst.dtype() )
  {
    TORCH_CHECK ( src.nbytes() == dst.nbytes() );
    if ( src.device().type() == DeviceType::CPU && dst.device().type() == DeviceType::PrivateUse1 )
    {
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), src.data_ptr(), dst.nbytes() );
    }
    else if ( src.device().type() == DeviceType::PrivateUse1 && dst.device().type() == DeviceType::CPU )
    {
      tpu::TPUCopyDeviceToHost ( dst.data_ptr(), src.data_ptr(), dst.nbytes() );
    }
    else if ( src.device().type() == DeviceType::PrivateUse1 && dst.device().type() == DeviceType::PrivateUse1 )
    {
      tpu::TPUCopyDeviceToDevice ( dst.data_ptr(), src.data_ptr(), dst.nbytes() );
    }
    else
    {
      LOG ( FATAL ) << "Unsupported copy from device " << src.device()
                    << " to device " << dst.device();
    }
  }
  else
  {
    if ( src.device().type() == DeviceType::CPU && dst.device().type() == DeviceType::PrivateUse1 )
    {
      _copy_from_tpu ( src.to ( dst.device() ), dst, non_blocking );
    }
    else if ( src.device().type() == DeviceType::PrivateUse1 &&  dst.device().type() == DeviceType::CPU )
    {
      _copy_from_tpu ( src.to ( dst.dtype() ), dst, non_blocking );
    }
    else if ( src.device().type() == DeviceType::PrivateUse1 && dst.device().type() == DeviceType::PrivateUse1 )
    {
      LOG ( FATAL ) << "Cast from " << src.dtype()
                    << " to " << dst.dtype() << " is not implemented";
    }
    else
    {
      LOG ( FATAL ) << "Unsupported copy from device " << src.device()
                    << " to device " << dst.device();
    }
  }
  return dst;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_copy_from", _copy_from_tpu );
}
} // namespace at
