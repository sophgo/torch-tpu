#pragma once

#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <sgdnn_api.h>
#include <TPUDeviceManager.h>
#include <sys/time.h>
#include <cmath>
#include <mutex>

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
                                      const at::Tensor & Exp,
                                      double ErrScale = 1.0 )
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
    int ErrCnt = 0;
    const int MaxErrCnt = 100;
    auto AbsErr = torch::abs ( torch::sub ( Got, Exp ) );
    auto AbsExp = torch::abs ( Exp );
    auto RltAbsErr = torch::div ( AbsErr, AbsExp ) * ErrScale;
    auto GotPtr = Got.data_ptr<float>();
    auto ExpPtr = Exp.data_ptr<float>();
    auto AbsErrPtr = AbsErr.data_ptr<float>();
    auto RltAbsErrPtr = RltAbsErr.data_ptr<float>();
    for ( auto i = 0; i < Got.numel(); ++i )
    {
      if ( std::isnan ( ExpPtr[i] ) == true && std::isnan ( GotPtr[i] ) == true )
      {
        continue;
      }
      if ( AbsErrPtr[i] < 1e-4 || RltAbsErrPtr[i] <= 1e-5 )
      {
        continue;
      }
FAILED:
      if ( ErrCnt < MaxErrCnt )
      {
        LOG ( WARNING ) << "Compare failed: Got = " << GotPtr[i]
                        << ", Exp = " << ExpPtr[i]
                        << ", index = " << i;
      }
      else
      {
        LOG ( WARNING ) << "<Skip the other compare errors>";
        return;
      }
      ++ErrCnt;
    }
  }
  else
  {
    LOG ( FATAL ) << "Unsupported data type " << Got.dtype();
  }
}

struct Timer
{
  Timer & Start()
  {
    gettimeofday ( &timer, NULL );
    return *this;
  }
  unsigned long ElapsedUS()
  {
    struct timeval end;
    gettimeofday ( &end, NULL );
    return ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec );
  }
  unsigned long ElapsedMS()
  {
    return ElapsedUS() / 1000;
  }
private:
  struct timeval timer;
};

typedef enum
{
  CONVOLUTION = 0,
  CONVOLUTION_BACKWARD,
  BATCHNORM,
  BATCHNORM_BACKWARD,
  RELU,
  RELU_BACKWARD,
  OP_NUM
}
OpType;

static const char * OpTypeStr[OP_NUM] =
{
  "Convolution",
  "Convolution Backward",
  "Batchnorm",
  "Batchnorm Backward",
  "ReLU",
  "ReLU Backward"
};

struct OpTimer
{
  OpTimer & Clear();
  OpTimer & AddTime ( OpType type, unsigned long time_us );
  void Dump() const;
  static OpTimer & Instance();
private:
  unsigned long long elapsed_time_us_[OP_NUM];
  std::mutex mutex_;
  static OpTimer * instance_;
};

} // namespace tpu
