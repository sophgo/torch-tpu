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
#include <fstream>
#include <vector>

#define CHECK_TENSOR_IN_DEVICE(t) \
do \
{ \
TORCH_CHECK ( t.device().type() == DeviceType::TPU, #t, " is not in TPU device" ); \
TORCH_CHECK ( t.is_contiguous() == true, __FILE__ ,":" ,__func__, ": ", #t, " is not contiguous" ); \
} \
while ( 0 )

#define CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(t) \
do \
{ \
TORCH_CHECK ( t.device().type() == DeviceType::TPU, #t, " is not in TPU device" ); \
} \
while ( 0 )

#define TENSOR_TO_CPU(t) ( ( t ).to ( torch::Device ( "cpu" ) ) )

#define TENSOR_TO_TPU(t) ( ( t ).to ( tpu::TPUGetCurrentDevice() ) )

#define ADDR_IN_DEVICE(t) (t).data_ptr()

#define TPU_ERROR_CODE(Err) " ( TPU error code: " << Err << ")"

#define IS_TPU_TENSOR(t)  ( ( t ).device().type() == DeviceType::TPU )
#define IS_CPU_TENSOR(t)  ( ( t ).device().type() == DeviceType::CPU )

namespace tpu
{

static inline at::Device TPUGetCurrentDevice()
{
  return at::Device ( at::DeviceType::TPU, tpu::TPUGetDeviceIndex() );
}

static inline void SaveTensorToBinaryFile ( const at::Tensor & Tensor, const std::string & Path )
{
  auto TensorCPU = TENSOR_TO_CPU ( Tensor ).contiguous();
  std::ofstream fout ( Path, std::ios::out | std::ios::binary );
  TORCH_CHECK ( fout.is_open() == true, "Failed to open file ", Path );
  fout.write ( ( char * ) TensorCPU.data_ptr(), TensorCPU.nbytes() );
  fout.close();
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
  else if ( dtype == caffe2::TypeMeta::Make<int>() )
  {
    return 6;
  }
  else if ( dtype == caffe2::TypeMeta::Make<bool>() ||
            dtype == caffe2::TypeMeta::Make<unsigned char>() ) {
    return 3;
  }
  else if ( dtype == caffe2::TypeMeta::Make<long>() )
  {
    return 31;
  }
  else
  {
    TORCH_CHECK ( false, "Unsupported data type ", dtype );
  }
  return -1;
}

static inline TensorDescriptor_t TPUGenerateTensorDesc ( const at::Tensor & Tensor )
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
                                      double Threshold = 1e-4,
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
    const auto MaxErrCnt = 100;//Got.numel();
    auto Err = torch::sub ( Got, Exp );
    auto AbsErr = torch::abs ( Err );
    auto AbsExp = torch::abs ( Exp );
    auto RltAbsErr = torch::div ( AbsErr, AbsExp ) * ErrScale;
    auto ErrPtr = Err.data_ptr<float>();
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
      if ( AbsErrPtr[i] < Threshold || RltAbsErrPtr[i] <= 1e-5 )
      {
        continue;
      }
FAILED:
      if ( ErrCnt < MaxErrCnt )
      {
        LOG ( WARNING ) << "Compare failed: Got = " << GotPtr[i]
                        << ", Exp = " << ExpPtr[i]
                        << ", Err = " << ErrPtr[i]
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
  unsigned long ElapsedUS() const
  {
    struct timeval end;
    gettimeofday ( &end, NULL );
    return ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec );
  }
  unsigned long ElapsedMS() const
  {
    return ElapsedUS() / 1000;
  }
private:
  struct timeval timer;
};

typedef enum
{
  CDMA_D2S = 0,
  CDMA_S2D,
  CDMA_C2C,
  COPY,
  CONVOLUTION,
  CONVOLUTION_BACKWARD,
  BATCHNORM,
  BATCHNORM_BACKWARD,
  LAYERNORM,
  LAYERNORM_BACKWARD,
  RELU,
  RELU_BACKWARD,
  GELU,
  GELU_BACKWARD,
  MM,
  ADDMM,
  BMM,
  LINEAR,
  SOFTMAX,
  SOFTMAX_BACKWARD,
  PERMUTE,
  TRANSPOSE,
  ADD,
  SUB,
  MUL,
  DIV,
  INDEX_SELECT,
  DTYPE_CONVERT,
  REDUCE_MEAN,
  REDUCE_SUM,
  WHERE,
  STRIDED_COPY,
  CONCAT,
  CONST_FILL,
  SQRT,
  ADDCDIV,
  ADDCMUL,
  EMBEDDING_BACKWARD,
  MALLOC,
  FREE,
  CROSS_ENTROPY_LOSS,
  CROSS_ENTROPY_LOSS_BACKWARD,
  SCALE_ADD,
  OP_NUM
}
OpType;

static const char * OpTypeStr[OP_NUM] =
{
  "CDMA D2S",
  "CDMA S2D",
  "CDMA C2C",
  "Copy",
  "Convolution",
  "Convolution Backward",
  "BatchNorm",
  "BatchNorm Backward",
  "LayerNorm",
  "LayerNorm Backward",
  "ReLU",
  "ReLU Backward",
  "GeLU",
  "GeLU Backward",
  "MatMul",
  "Add MatMul",
  "Batch MatMul",
  "Linear",
  "Softmax",
  "Softmax Backward",
  "Permute",
  "Transpose",
  "Add",
  "Sub",
  "Mul",
  "Div",
  "Index Select",
  "DType Convert",
  "Reduce Mean",
  "Reduce Sum",
  "Where",
  "Strided Copy",
  "Concat",
  "Const Fill",
  "Sqrt",
  "Addcdiv",
  "Addcmul",
  "Embedding Backward",
  "Malloc",
  "Free",
  "Cross Entropy Loss",
  "Cross Entropy Loss Backward",
  "Scale Add"
};

struct OpTimer
{
  OpTimer & Clear();
  OpTimer & Start();
  OpTimer & Pause();
  OpTimer & AddTime ( OpType type, unsigned long time_us );
  void Dump() const;
  static OpTimer & Instance();
private:
  OpTimer() {}
  unsigned long long elapsed_time_us_[OP_NUM];
  bool is_paused_ = false;
  std::mutex mutex_;
  static OpTimer * instance_;
};

struct GlobalTimer
{
  GlobalTimer & Reset();
  void Dump() const;
  static GlobalTimer & Instance();
private:
  GlobalTimer() {}
  Timer timer_;
  static GlobalTimer * instance_;
};

struct TensorWatcher
{
  void AddTensor ( const at::Tensor & Tensor );
  bool Watch() const;
  static TensorWatcher & Instance();
private:
  TensorWatcher() {}
  std::vector<at::Tensor> tensors_;
  std::vector<at::Tensor> tensors_cpu_;
  static TensorWatcher * instance_;
};

} // namespace tpu
