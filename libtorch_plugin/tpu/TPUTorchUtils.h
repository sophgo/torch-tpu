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

#ifdef TPU_OP_TIMING
#define TIMING_START auto timer = tpu::Timer().Start();
#define TIMING_END(TIMING_NAME)                                                \
  tpu::OpTimer::Instance().AddTime(TIMING_NAME, timer.ElapsedUS());
#else
#define TIMING_START
#define TIMING_END(OP)
#endif

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

static inline SgdnnDataType_t TPUConvertDType ( caffe2::TypeMeta dtype )
{
  if ( dtype == caffe2::TypeMeta::Make<float>() )
  {
    return SGDNN_DTYPE_FP32;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::Half>() )
  {
    return SGDNN_DTYPE_FP16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::BFloat16>() )
  {
    return SGDNN_DTYPE_BF16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<long>() )
  {
    return SGDNN_DTYPE_INT64;
  }
  else if ( dtype == caffe2::TypeMeta::Make<int>() )
  {
    return SGDNN_DTYPE_INT32;
  }
  else if ( dtype == caffe2::TypeMeta::Make<bool>() ||
            dtype == caffe2::TypeMeta::Make<unsigned char>() ) {
    return SGDNN_DTYPE_UINT8;
  }
  else if ( dtype == caffe2::TypeMeta::Make<int8_t>() )
  {
    return SGDNN_DTYPE_INT8;
  }
  else
  {
    TORCH_CHECK ( false, "Unsupported data type ", dtype );
  }
  return SGDNN_DTYPE_UNKNOWN;
}

static inline SgdnnTensor_t TPUGenerateSgdnnTensor ( const at::Tensor & Tensor )
{
  SgdnnTensor_t t = { 0 };
  t.addr = ( unsigned long long ) Tensor.data_ptr();
  t.dtype = TPUConvertDType ( Tensor.dtype() );
  t.dim = Tensor.dim();
  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    t.shape[i] = Tensor.size ( i );
    t.stride[i] = Tensor.stride ( i );
  }
  return t;
}

static inline bool TPUIsSameShape ( const at::Tensor & Tensor1, const at::Tensor & Tensor2 )
{
  if ( Tensor1.dim() == Tensor2.dim() )
  {
    for ( auto i = 0; i < Tensor1.dim(); ++i )
    {
      if ( Tensor1.size ( i ) != Tensor2.size ( i ) )
      {
        return false;
      }
    }
  }
  else
  {
    return false;
  }
  return true;
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
  LEAKY_RELU,
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
  REDUCE_PROD,
  WHERE,
  STRIDED_COPY,
  CONCAT,
  CONST_FILL,
  SQRT,
  RSQRT,
  SIGN,
  ADDCDIV,
  ADDCMUL,
  EMBEDDING_BACKWARD,
  MALLOC,
  FREE,
  CROSS_ENTROPY_LOSS,
  CROSS_ENTROPY_LOSS_BACKWARD,
  ADD_C,
  MUL_C,
  C_SUB,
  C_DIV,
  NORM2,
  BCAST_ADD,
  MLP_FORWARD,
  MLP_BACKWARD,
  ATTN_FORWARD,
  ATTN_BACKWARD,
  BITWISE_XOR,
  BITWISE_XOR_BCAST,
  BITWISE_XOR_C,
  ABS_FORWARD,
  COS_FORWARD,
  SIN_FORWARD,
  TAN_FORWARD,
  LOG_FORWARD,
  ACOSH_FORWARD,
  ASINH_FORWARD,
  ATANH_FORWARD,
  SINH_FORWARD,
  COSH_FORWARD,
  TANH_FORWARD,
  EXP_FORWARD,
  ASIN,
  ACOS,
  ATAN,
  SINH,
  COSH,
  TANH,
  CEIL,
  FLOOR,
  ROUND,
  NEG,
  EXP2,
  EXPM1,
  SQUEEZE,
  UNSQUEEZE,
  ISFINITE,
  ISINF,
  ISNAN,
  BITWISE_NOT,
  MINIMUM,
  MAXIMUM,
  FMIN,
  FMAX,
  ATAN2,
  LOGICAL_AND, 
  LOGICAL_OR,
  BITWISE_AND,
  BITWISE_AND_BCAST,
  BITWISE_AND_C,
  BITWISE_OR,
  BITWISE_OR_BCAST,
  BITWISE_OR_C,
  EQUAL,
  EQUAL_BCAST,
  EQUAL_C,
  GREATER_OR_EQUAL,
  GREATER_OR_EQUAL_BCAST,
  GREATER_OR_EQUAL_C,
  GREATER,
  GREATER_BCAST,
  GREATER_C,
  LESS_THAN_OR_EQUAL,
  LESS_THAN_OR_EQUAL_BCAST,
  LESS_THAN_OR_EQUAL_C,
  SHIFT_LEFT,
  SHIFT_LEFT_BCAST,
  SHIFT_LEFT_C,
  SHIFT_RIGHT_ARITHMETIC,
  SHIFT_RIGHT_ARITHMETIC_BCAST,
  SHIFT_RIGHT_ARITHMETIC_C,
  LESS_THAN,
  LESS_THAN_BCAST,
  LESS_THAN_C,
  NOT_EQUAL,
  NOT_EQUAL_BCAST,
  NOT_EQUAL_C,
  SIGNBIT,
  FULL,
  LOGICAL_NOT,
  ARANGE,
  SILU,
  SIGMOID,
  CLAMP,
  LN_MM_FORWARD,
  UPSAMPLING_BILINEAR,
  ERF,
  POW_FORWARD,
  ADD_LN_MM_FORWARD,
  RECIPROCAL,
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
  "Leaky ReLU",
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
  "REDUCE_PROD",
  "Where",
  "Strided Copy",
  "Concat",
  "Const Fill",
  "Sqrt",
  "Rsqrt",
  "Sign",
  "Addcdiv",
  "Addcmul",
  "Embedding Backward",
  "Malloc",
  "Free",
  "Cross Entropy Loss",
  "Cross Entropy Loss Backward",
  "AddC",
  "MulC",
  "CSub",
  "CDiv",
  "Norm2",
  "Bcast Add",
  "MLP Forward",
  "MLP Backward",
  "Attention Forward",
  "Attention Backward",
  "BITWISE_XOR",
  "BITWISE_XOR_BCAST",
  "BITWISE_XOR_C",
  "Abs Forward",
  "Cos Forward",
  "Sin Forward",
  "Tan Forward",
  "Log Forward",
  "ACosH Forward",
  "ASinH Forward",  
  "ATanH Forward",
  "SinH Forward",
  "CosH Forward",
  "TanH Forward",
  "Exp Forward",
  "Asin",
  "Acos",
  "Atan",
  "Sinh",
  "Cosh",
  "Tanh",
  "Ceil",
  "Floor",
  "Round",
  "Neg",
  "Exp2",
  "EXPM1",
  "Squeeze",
  "Unsqueeze",
  "Isfinite",
  "Isinf",
  "Isnan",
  "Bitwise_Not",
  "Minimum",
  "Maximum",
  "Fmin",
  "Fmax",
  "Atan2",
  "Logical And",
  "Logical Or",
  "BITWISE_AND",
  "BITWISE_AND_BCAST",
  "BITWISE_AND_C",
  "BITWISE_OR",
  "BITWISE_OR_BCAST",
  "BITWISE_OR_C",
  "EQUAL",
  "EQUAL_BCAST",
  "EQUAL_C",
  "GREATER_OR_EQUAL",
  "GREATER_OR_EQUAL_BCAST",
  "GREATER_OR_EQUAL_C",
  "GREATER",
  "GREATER_BCAST",
  "GREATER_C",
  "LESS_THAN_OR_EQUAL",
  "LESS_THAN_OR_EQUAL_BCAST",
  "LESS_THAN_OR_EQUAL_C",
  "SHIFT_LEFT",
  "SHIFT_LEFT_BCAST",
  "SHIFT_LEFT_C",
  "SHIFT_RIGHT_ARITHMETIC",
  "SHIFT_RIGHT_ARITHMETIC_BCAST",
  "SHIFT_RIGHT_ARITHMETIC_C",
  "LESS_THAN",
  "LESS_THAN_BCAST",
  "LESS_THAN_C",
  "NOT_EQUAL",
  "NOT_EQUAL_BCAST",
  "NOT_EQUAL_C",
  "SIGNBIT",
  "FULL",
  "Logical Not",
  "Arrange",
  "SiLU",
  "Sigmoid",
  "CLAMP",
  "layernorm Matmul",
  "ERF",
  "Pow Forward",
  "Add Layernorm Matmul",
  "RECIPROCAL"
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
