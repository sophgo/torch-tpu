#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <sgdnn_api.h>

#define CHECK_TENSOR_IN_DEVICE(t) \
do                                                                 \
{                                                                  \
if ( tpu::TPUPtrIsInCurrentDevice ( t.data_ptr() ) == false )      \
{                                                                  \
  LOG ( FATAL ) << #t << " is not in current device";              \
}                                                                  \
}                                                                  \
while ( 0 )

#define ADDR_IN_DEVICE(t) tpu::TPUGetAddrInDevice( t.data_ptr() )

namespace tpu
{

static inline TensorDescriptor_t TPUGenerateTensorDesc (
const at::Tensor & Tensor )
{
  TensorDescriptor_t Desc = { 0 };
  if ( Tensor.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    Desc.dtype = 1;
  }
  else
  {
    LOG ( FATAL ) << "Unsupported data type " << Tensor.dtype();
  }
  Desc.ndims = Tensor.dim();
  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    Desc.shape[i] = Tensor.size ( i );
    Desc.stride[i] = Tensor.stride ( i );
  }
  return Desc;
}

} // namespace tpu
