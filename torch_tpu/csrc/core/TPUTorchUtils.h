#pragma once

#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <c10/util/Half.h>
#include <sgdnn_api.h>
#include <TPUDeviceManager.h>
#include <sys/time.h>
#include <cmath>
#include <mutex>
#include <fstream>
#include <vector>

#define DEBUG_SHOW(dtype, func, ...)                    \
do{                                                     \
  std::cout << __func__ << "|" << #__VA_ARGS__ << "";   \
  const dtype args[] = {__VA_ARGS__};                   \
  int numArgs = sizeof(args) / sizeof(args[0]);         \
  for (int i = 0; i < numArgs; ++i) {                   \
      std::cout << "| " << func(args[i]);               \
  }                                                     \
  std::cout << "\n";                                    \
} while (0)

#ifdef TPU_OP_TIMING
#define TIMING_START auto timer = tpu::Timer().Start();
#define TIMING_END(TIMING_NAME)  \
  tpu::OpTimer::Instance().AddTime(TIMING_NAME, timer.ElapsedUS());
#else
#define TIMING_START
#define TIMING_END(OP)
#endif

#ifdef SHOW_OP_INFO
#define SHOW_TENSOR_OP(...) DEBUG_SHOW(Tensor, tpu::GetTensorInfo, __VA_ARGS__)
#define SHOW_EMPTY_INFO(Tensor) std::cout << __func__ << ": " << Tensor.data_ptr() << std::endl;
#else
#define SHOW_TENSOR_OP(...) 
#define SHOW_EMPTY_INFO(Tensor)
#endif

#ifdef SHOW_CPU_OP
#define CPU_IMPL_WANING(...)  LOG( WARNING ) << __func__ << " use cpu impl." << #__VA_ARGS__;
#else
#define CPU_IMPL_WANING(...)
#endif

#define CHECK_TENSOR_IN_DEVICE(t) \
do \
{ \
TORCH_CHECK ( t.device().type() == DeviceType::TPU, ":", __func__, ": ", #t, " is not in TPU device" ); \
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

#define ADDR_IN_DEVICE(t) GetAddrByUnifiedAddr((unsigned long long)(t).data_ptr())

#define TPU_ERROR_CODE(Err) " ( TPU error code: " << Err << ")"

#define IS_TPU_TENSOR(t)  ( ( t ).device().type() == DeviceType::TPU )
#define IS_CPU_TENSOR(t)  ( ( t ).device().type() == DeviceType::CPU )

#define TPU_DEVICE_INDEX_BITS 6
#define TPU_GLOBAL_ADDR_BITS (64 - TPU_DEVICE_INDEX_BITS)


namespace tpu
{

static inline unsigned long long UnifiedAddr( unsigned long long Addr, int Index)
{
  TORCH_CHECK ( Addr < ( 1UL << TPU_GLOBAL_ADDR_BITS ) );
  return ( ( ( unsigned long long ) Index ) << TPU_GLOBAL_ADDR_BITS ) | Addr;
}

static inline unsigned long long GetDeviceIndexByUnifiedAddr ( unsigned long long Addr )
{
  return Addr >> TPU_GLOBAL_ADDR_BITS;
}

static inline unsigned long long GetAddrByUnifiedAddr ( unsigned long long Addr )
{
  return ( Addr << TPU_DEVICE_INDEX_BITS ) >> TPU_DEVICE_INDEX_BITS;
}

static inline at::Device TPUGetCurrentDevice()
{
  return at::Device ( at::DeviceType::TPU, tpu::TPUGetDeviceIndex() );
}

static inline void SaveTensorToBinaryFile ( const at::Tensor & Tensor, const std::string & Path )
{
  if (!Tensor.has_storage()) { return; }
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
  else if (dtype == caffe2::TypeMeta::Make<short>()){
    return SGDNN_DTYPE_INT16;
  }
  else
  {
    TORCH_CHECK ( false, "Unsupported data type ", dtype );
  }
  return SGDNN_DTYPE_UNKNOWN;
}

static inline bool IsSupportDtype( caffe2::TypeMeta&& dtype )
{
  bool support = false;
  if ( dtype == caffe2::TypeMeta::Make<float>() || dtype == caffe2::TypeMeta::Make<at::Half>() ||
       dtype == caffe2::TypeMeta::Make<at::BFloat16>() || dtype == caffe2::TypeMeta::Make<int>() ||
       dtype == caffe2::TypeMeta::Make<bool>() || dtype == caffe2::TypeMeta::Make<unsigned char>() ||
       dtype == caffe2::TypeMeta::Make<int8_t>())
  {
    support = true;
  }
  return support;
}

static inline SgdnnTensor_t TPUGenerateSgdnnTensor ( const at::Tensor & Tensor )
{
  SgdnnTensor_t t = { 0 };
  t.addr = GetAddrByUnifiedAddr(( unsigned long long ) Tensor.data_ptr());
  t.dtype = TPUConvertDType ( Tensor.dtype() );
  t.dim = Tensor.dim();
  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    t.shape[i] = Tensor.size ( i );
    t.stride[i] = Tensor.stride ( i );
  }
  return t;
}

static inline std::string GetTensorInfo( const at::Tensor & Tensor )
{
  std::ostringstream Tensor_info;
  if (Tensor.has_storage()){
    auto dtype = Tensor.dtype();
    Tensor_info << "addr : " << Tensor.data_ptr() << ", ";
    Tensor_info << "dtype : " << dtype << ", ";
    if (dtype == caffe2::TypeMeta::Make<float>())
    {
      Tensor_info << "data0 : " << *((float*)(Tensor.cpu().data_ptr())) << ",";
    }
    else if (dtype == caffe2::TypeMeta::Make<c10::Half>())
    {
      Tensor_info << "data0 : " << (float)c10::detail::fp16_ieee_to_fp32_value(((c10::Half*)(Tensor.cpu().data_ptr()))->x) << ",";
    }
    Tensor_info << "size : [";
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      Tensor_info << " " << Tensor.size ( i );
    }
    Tensor_info << "], strides : [";
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      Tensor_info << " " << Tensor.stride ( i );
    }
    Tensor_info << "];";
  }
  return Tensor_info.str();
}

static inline SgdnnTensor_t TPUGenerateSgdnnTensorforComplex64 ( const at::Tensor & Tensor )
{
  SgdnnTensor_t t = { 0 };
  t.addr = GetAddrByUnifiedAddr(( unsigned long long ) Tensor.data_ptr());
  t.dtype = TPUConvertDType ( caffe2::TypeMeta::Make<float>() );
  t.dim = Tensor.dim();
  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    t.shape[i] = Tensor.size ( i );
    t.stride[i] = Tensor.stride ( i ) * 2;
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
  AVG_POOLING,
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
  ADDBCAST,
  INDEX_SELECT,
  DTYPE_CONVERT,
  REDUCE_MEAN,
  REDUCE_SUM,
  REDUCE_PROD,
  REDUCE_MAX,
  REDUCE_MIN,
  REDUCE_VAR,
  WHERE,
  STRIDED_COPY,
  CONCAT,
  CONST_FILL,
  MASKED_FILL,
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
  NATIVE_GROUP_NORM,
  GROUPNORM_BACKWARD,
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
  EXPAND,
  FLIP,
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
  UPSAMPLING_BILINEAR,
  UPSAMPLING_NEAREST,
  UPSAMPLING_NEAREST_BACKWARD,
  ARANGE,
  SILU,
  SIGMOID,
  CLAMP,
  LN_MM_FORWARD,
  ERF,
  ERFC,
  POW_FORWARD,
  ADD_LN_MM_FORWARD,
  RECIPROCAL,
  TRUNC,
  POWC,
  LN_MM_BACKWARD,
  ADD_LN_MM_BACKWARD,
  TOPK,
  NONZERO,
  REPEAT,
  ARGMAX,
  ARGMIN,
  MAX_DIM,
  MIN_DIM,
  HARDTANH,
  HYPOT,
  NEXTAFTER,
  TRIU,
  CBRT,
  CONSTANT_PAD,
  REFLECTION_PAD2D,
  REPLICATION_PAD2D,
  REPLICATION_PAD3D,
  GATHER,
  BADDBMM,
  MSE_LOSS,
  MSE_LOSS_BACKWARD,
  SLICE_SCATTER,
  InfCheckAndUnscale,
  LLAMA_ATTENTION,
  LLAMA_MLP_FORWARD,
  RMSNORM_FORWARD,
  BINARYOP,
  BINARYOP_C,
  BINARYOP_BCAST,
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
  "Avg Pooling",
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
  "Add Bcast",
  "Index Select",
  "DType Convert",
  "Reduce Mean",
  "Reduce Sum",
  "REDUCE_PROD",
  "Reduce Max",
  "Reduce Min",
  "Reduce var",
  "Where",
  "Strided Copy",
  "Concat",
  "Const Fill",
  "Masked Fill",
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
  "Native Group Norm",
  "Groupnorm backward",
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
  "Expand",
  "Flip",
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
  "Upsampl Bilinear2d",
  "Upsampl nearest",
  "Upsampl nearest2d Backward",
  "Arrange",
  "SiLU",
  "Sigmoid",
  "CLAMP",
  "layernorm Matmul",
  "ERF",
  "ERFC",
  "Pow Forward",
  "Add Layernorm Matmul",
  "RECIPROCAL",
  "TRUNC",
  "POW SCALAR",
  "layernorm Matmul Backward",
  "Add layernorm Matmul Backward",
  "TOPK",
  "NonZero",
  "REPEAT",
  "Argmax",
  "Argmin",
  "Max_dim",
  "Min_dim",
  "HARDTANH",
  "HYPOT",
  "Nextafter",
  "Triu",
  "Cbrt",
  "Constant_pad",
  "Reflection_pad2d",
  "Replication_pad2d",
  "Replication_pad3d",
  "GATHER",
  "BADDBMM",
  "MSE Loss",
  "MSE Loss Backward",
  "Slice_scatter",
  "Inf Check And Unscale",
  "LLAMA_MLP_FORWARD",
  "RMSNORM_FORWARD",
  "binary_op",
  "binary_op_c",
  "binary_op_bcast",
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
