#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <sgdnn_api.h>
#include <TPUDeviceManager.h>

#define CHECK_TENSOR_IN_DEVICE(t)                                  \
do                                                                 \
{                                                                  \
if ( t.device().type() != DeviceType::PrivateUse1 )                \
{                                                                  \
  LOG ( FATAL ) << #t << " is not in TPU device";                  \
}                                                                  \
if ( tpu::TPUPtrIsInCurrentDevice ( t.data_ptr() ) == false )      \
{                                                                  \
  LOG ( FATAL ) << #t << " is not in current TPU device";          \
}                                                                  \
}                                                                  \
while ( 0 )

#define ADDR_IN_DEVICE(t) tpu::TPUGetAddrInDevice( t.data_ptr() )

#define TPU_ERROR_CODE(Err) " ( TPU error code: " << Err << ")"

namespace tpu
{

static inline at::Device TPUGetCurrentDevice()
{
  return at::Device ( at::DeviceType::PrivateUse1, tpu::TPUGetDeviceIndex() );
}

static inline int TPUConvertDType ( caffe2::TypeMeta dtype )
{
  if ( dtype == caffe2::TypeMeta::Make<float>() )
  {
    return 0;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::Half>() )
  {
    return 1;
  }
  else
  {
    LOG ( FATAL ) << "Unsupported data type " << dtype;
  }
  return -1;
}

static inline TensorDescriptor_t TPUGenerateTensorDesc (
const at::Tensor & Tensor )
{
  TensorDescriptor_t Desc = { 0 };
  Desc.dtype = TPUConvertDType ( Tensor.dtype() );
  Desc.ndims = Tensor.dim();
  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    Desc.shape[i] = Tensor.size ( i );
    Desc.stride[i] = Tensor.stride ( i );
  }
  return Desc;
}

static inline void TPUCompareResult ( const at::Tensor & Got,
                                      const at::Tensor & Exp )
{
  if ( Got.dtype() != Exp.dtype() )
  {
    LOG ( FATAL ) << "Tensor comparing failed: Got data type = "
                  << Got.dtype() << ", Exp data type = " << Exp.dtype();
  }
  if ( Got.sizes() != Exp.sizes() )
  {
    LOG ( FATAL ) << "Tensor comparing failed: Got shape = "
                  << Got.sizes() << ", Exp shape = " << Exp.sizes();
  }
  if ( Got.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    auto AbsErr = torch::abs ( torch::sub ( Got, Exp ) );
    auto AbsExp = torch::abs ( Exp );
    auto RltAbsErr = torch::div ( AbsErr, AbsExp );
    auto GotPtr = Got.data_ptr<float>();
    auto ExpPtr = Exp.data_ptr<float>();
    auto AbsErrPtr = AbsErr.data_ptr<float>();
    auto AbsExpPtr = AbsExp.data_ptr<float>();
    auto RltAbsErrPtr = RltAbsErr.data_ptr<float>();
    for ( auto i = 0; i < Got.numel(); ++i )
    {
      if ( AbsErrPtr[i] == 0.f )
      {
        continue;
      }
      if ( AbsExpPtr[i] == 0.f && AbsErrPtr[i] == 0.f )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && RltAbsErrPtr[i] <= 1e-2 )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && AbsExpPtr[i] < 1e-3 &&
           RltAbsErrPtr[i] <= 0.1 )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && AbsExpPtr[i] < 1e-4 &&
           RltAbsErrPtr[i] <= 0.6 )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && AbsExpPtr[i] < 1e-5 &&
           RltAbsErrPtr[i] <= 2.5 )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && AbsExpPtr[i] < 1e-6 &&
           RltAbsErrPtr[i] <= 170 )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && AbsExpPtr[i] < 1e-7 &&
           RltAbsErrPtr[i] <= 200 )
      {
        continue;
      }
      if ( AbsExpPtr[i] != 0.f && AbsExpPtr[i] < 1e-8 &&
           RltAbsErrPtr[i] <= 250 )
      {
        continue;
      }
      LOG ( FATAL ) << "Tensor comparing failed: Got = " << GotPtr[i]
                    << ", Exp = " << ExpPtr[i]
                    << ", RltAbsErr = " << RltAbsErrPtr[i]
                    << ", index = " << i;
    }
  }
  else
  {
    LOG ( FATAL ) << "Unsupported data type " << Got.dtype();
  }
}

} // namespace tpu
