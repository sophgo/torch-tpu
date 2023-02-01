#include <torch/library.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <c10/util/Logging.h>

#include <iostream>

namespace at
{
Tensor _copy_from_tpu ( const Tensor & self, const Tensor & dst,
                        bool non_blocking )
{
  if ( self.is_contiguous() == false || dst.is_contiguous() == false )
  {
    LOG ( FATAL ) << "TPU only supports contiguous memory copy for now";
  }
  if ( self.nbytes() != dst.nbytes() )
  {
    LOG ( FATAL ) << "Sizes of src and dst are different";
  }
  if ( self.device().type() == DeviceType::CPU &&
       dst.device().type() == DeviceType::PrivateUse1 )
  {
    tpu::TPUCopyHostToDevice ( dst.data_ptr(), self.data_ptr(), dst.nbytes() );
  }
  else if ( self.device().type() == DeviceType::PrivateUse1 &&
            dst.device().type() == DeviceType::CPU )
  {
    tpu::TPUCopyDeviceToHost ( dst.data_ptr(), self.data_ptr(), dst.nbytes() );
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
  auto tensor = dst;
  return tensor;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_copy_from", _copy_from_tpu );
}
} // namespace at
