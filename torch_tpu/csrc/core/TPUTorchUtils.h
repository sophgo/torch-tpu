#pragma once
#include <sys/time.h>
#include <cmath>
#include <mutex>
#include <fstream>
#include <vector>
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <c10/util/Half.h>
#include <type_traits>

#include <Python.h>
#include <frameobject.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include "TPUDeviceManager.h"
#include "TPUGuard.h"
#include <tpuDNN.h>
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"
#include "TPUStream.h"

#ifdef BACKEND_SG2260
#include <sgdnn_api.h>
#elif defined BACKEND_1684X
#include <sgdnn_api.h>
#endif

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
#define SHOW_TENSOR_OP(...) DEBUG_SHOW(at::Tensor, tpu::GetTensorInfo, __VA_ARGS__)
#define SHOW_EMPTY_INFO(Tensor) std::cout << __func__ << ": " << Tensor.data_ptr() << std::endl;
#else
#define SHOW_TENSOR_OP(...)
#define SHOW_EMPTY_INFO(Tensor)
#endif

#ifdef SHOW_CPU_OP
#define CPU_IMPL_WARNING(...)  LOG( WARNING ) << __func__ << " use cpu impl." << #__VA_ARGS__;
#else
#define CPU_IMPL_WARNING(...)
#endif

//#define PERF_MODE
#ifdef PERF_MODE

#define CHECK_TENSOR_IN_DEVICE(t)
#define CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(t)

#else

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

#endif

#define TENSOR_TO_CPU(t) ( ( t ).to ( torch::Device ( "cpu" ) ) )

#define TENSOR_TO_TPU(t) ( ( t ).to ( tpu::TPUGetCurrentDevice() ) )

#define ADDR_IN_DEVICE(t) GetAddrByUnifiedAddr((unsigned long long)(t).data_ptr())

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
  if (!Tensor.has_storage()) { return; }
  auto TensorCPU = TENSOR_TO_CPU ( Tensor ).contiguous();
  std::ofstream fout ( Path, std::ios::out | std::ios::binary );
  TORCH_CHECK ( fout.is_open() == true, "Failed to open file ", Path );
  fout.write ( ( char * ) TensorCPU.data_ptr(), TensorCPU.nbytes() );
  fout.close();
}

#define define_converter_entry(T, PREFIX, dtype) \
  const static T dtype = PREFIX##_##dtype;

#define define_converter(T, PREFIX)                   \
  define_converter_entry(T, PREFIX, DTYPE_FP32)       \
  define_converter_entry(T, PREFIX, DTYPE_FP16)       \
  define_converter_entry(T, PREFIX, DTYPE_BF16)       \
  define_converter_entry(T, PREFIX, DTYPE_INT64)      \
  define_converter_entry(T, PREFIX, DTYPE_INT32)      \
  define_converter_entry(T, PREFIX, DTYPE_UINT8)      \
  define_converter_entry(T, PREFIX, DTYPE_INT8)       \
  define_converter_entry(T, PREFIX, DTYPE_INT16)      \
  define_converter_entry(T, PREFIX, DTYPE_FP8E4M3)    \
  define_converter_entry(T, PREFIX, DTYPE_UNKNOWN)

template <typename T>
struct dtypes
{
  define_converter(SgdnnDataType_t, SGDNN)
};

template <>
struct dtypes<tpudnnDataType_t>
{
  define_converter(tpudnnDataType_t, TPUDNN)
};

#undef define_converter
#undef define_converter_entry

template <typename T>
static inline T TPUConvertDtype ( caffe2::TypeMeta dtype )
{
  if ( dtype == caffe2::TypeMeta::Make<float>() )
  {
    return dtypes<T>::DTYPE_FP32;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::Half>() )
  {
    return dtypes<T>::DTYPE_FP16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::BFloat16>() )
  {
    return dtypes<T>::DTYPE_BF16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<long>() )
  {
    return dtypes<T>::DTYPE_INT64;
  }
  else if ( dtype == caffe2::TypeMeta::Make<int>() )
  {
    return dtypes<T>::DTYPE_INT32;
  }
  else if ( dtype == caffe2::TypeMeta::Make<bool>() |
            dtype == caffe2::TypeMeta::Make<unsigned char>() ) {
    return dtypes<T>::DTYPE_UINT8;
  }
  else if ( dtype == caffe2::TypeMeta::Make<int8_t>() )
  {
    return dtypes<T>::DTYPE_INT8;
  }
  else if ( dtype == caffe2::TypeMeta::Make<short>() ) {
    return dtypes<T>::DTYPE_INT16;
  }
  else if ( dtype == caffe2::TypeMeta::Make<at::Float8_e4m3fn>() ) {
    return dtypes<T>::DTYPE_FP8E4M3;
  }
  else
  {
    TORCH_CHECK ( false, "Unsupported data type ", dtype );
  }
  return dtypes<T>::DTYPE_UNKNOWN;
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

template <typename T>
static inline T TPUGenerateDnnTensor ( const at::Tensor & Tensor )
{
  T t = { 0 };
  t.addr = reinterpret_cast<decltype(t.addr)>(
    GetAddrByUnifiedAddr(( unsigned long long ) Tensor.data_ptr()));
  t.dtype =TPUConvertDtype<decltype(t.dtype)>( Tensor.dtype() );
  t.dim = Tensor.dim();

  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    t.shape[i] = Tensor.size ( i );
    t.stride[i] = Tensor.stride ( i );
  }
  return t;
}
static inline SgdnnTensor_t TPUGenerateSgdnnTensor ( const at::Tensor & Tensor )
{
  SgdnnTensor_t t = { 0 };
  unsigned long long data_ptr;
  if (at_tpu::StorageDescHelper::IsBaseFormatType(Tensor)) {
    data_ptr = (unsigned long long)Tensor.data_ptr();
    t.dtype =TPUConvertDtype<decltype(t.dtype)>( Tensor.dtype() );
    t.dim = Tensor.dim();
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      t.shape[i] = Tensor.size ( i );
      t.stride[i] = Tensor.stride ( i );
    }
  }
  else {
    data_ptr = at_tpu::StorageDescHelper::GetDataPtrWithFormat(Tensor);
    at_tpu::StorageDescHelper::SetSgTensorAttributeWithFormat(Tensor, t);
   }
  t.addr = reinterpret_cast<decltype(t.addr)>( GetAddrByUnifiedAddr( data_ptr ) );
  return t;
}
using func_Sgdnn_t = SgdnnTensor_t (*)(const at::Tensor &);
// constexpr const static func_Sgdnn_t TPUGenerateSgdnnTensor = TPUGenerateDnnTensor<SgdnnTensor_t>;

static inline tpudnnTensor_t TPUGenerateTpudnnTensor(tpudnnHandle_t handle, const at::Tensor & Tensor)
{
  tpudnnTensor_t t = { 0 };
  unsigned long long data_ptr;
  if (at_tpu::StorageDescHelper::IsBaseFormatType(Tensor)) {
    data_ptr = (unsigned long long)Tensor.data_ptr();
    t.dtype =TPUConvertDtype<decltype(t.dtype)>( Tensor.dtype() );
    t.dim = Tensor.dim();
    for ( auto i = 0; i < Tensor.dim(); ++i )
    {
      t.shape[i] = Tensor.size ( i );
      t.stride[i] = Tensor.stride ( i );
    }
  }
  else {
    data_ptr = at_tpu::StorageDescHelper::GetDataPtrWithFormat(Tensor);
    at_tpu::StorageDescHelper::SettpuTensorAttributeWithFormat(Tensor, t);
   }
  if (Tensor.device().type() ==  DeviceType::TPU)
  {
    t.addr = reinterpret_cast<decltype(t.addr)>( GetAddrByUnifiedAddr( data_ptr ) );
    t.addr = tpudnnPhysToVirt(handle, (unsigned long long)t.addr);
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

template <typename T>
static inline T TPUGenerateTensorforComplex64 ( const at::Tensor & Tensor )
{
  T t = { 0 };
  t.addr = reinterpret_cast<decltype(t.addr)>(
    GetAddrByUnifiedAddr(( unsigned long long ) Tensor.data_ptr()));
  t.dtype = TPUConvertDtype<decltype(t.dtype)>( Tensor.dtype() );
  t.dim = Tensor.dim();
  for ( auto i = 0; i < Tensor.dim(); ++i )
  {
    t.shape[i] = Tensor.size ( i );
    t.stride[i] = Tensor.stride ( i ) * 2;
  }
  return t;
}

constexpr const static func_Sgdnn_t TPUGenerateSgdnnTensorforComplex64 = TPUGenerateDnnTensor<SgdnnTensor_t>;

static inline tpudnnTensor_t TPUGenerateTpudnnTensorforComplex64(tpudnnHandle_t handle, const at::Tensor & tensor)
{
  auto ret = TPUGenerateTensorforComplex64<tpudnnTensor_t>(tensor);
  ret.addr = tpudnnPhysToVirt(handle, (unsigned long long)ret.addr);
  return ret;
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

void TPUCompareResult ( const at::Tensor & Got, const at::Tensor & Exp,
                        double Threshold = 1e-4, double ErrScale = 1.0 );

typedef enum
{
  CDMA_D2S = 0,
  CDMA_S2D,
  CDMA_C2C,
  COPY,
  CPU_LAYER,
  CONVOLUTION,
  CONVOLUTION_BACKWARD,
  BATCHNORM,
  BATCHNORM_BACKWARD,
  LAYERNORM,
  LAYERNORM_BACKWARD,
  AVG_POOLING,
  MAX_POOLING,
  MAX_POOLING_WITH_MASK,
  MAX_POOLING_INDICES,
  RELU,
  RELU_BACKWARD,
  GELU,
  GELU_BACKWARD,
  LEAKY_RELU,
  LLama2A16MATMUL,
  MM,
  ADDMM,
  BMM,
  LINEAR,
  SOFTMAX,
  SOFTMAX_BACKWARD,
  LOGSOFTMAX,
  PERMUTE,
  TRANSPOSE,
  ADD,
  SUB,
  MUL,
  DIV,
  ADDBCAST,
  INDEX_SELECT,
  INDEX_ADD_,
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
  TANH_BACKWARD,
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
  SILU_BACKWARD,
  SIGMOID,
  SIGMOID_BACKWARD,
  CLAMP,
  LN_MM_FORWARD,
  ERF,
  ERFC,
  POW,
  POW_BCAST,
  POWC,
  CPOW,
  ADD_LN_MM_FORWARD,
  RECIPROCAL,
  TRUNC,
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
  LLAVA_MLP,
  LLAVA_ATTENTION,
  LLAMA_ATTENTION,
  LLAMA_ATTENTION_FORWARD,
  LLAMA_ATTENTION_BACKWARD,
  LLAMA_MLP_FORWARD,
  LLAMA_A16_MLP_FORWARD,
  RMSNORM_FORWARD,
  RMSNORM_BACKWARD,
  BINARYOP,
  BINARYOP_C,
  BINARYOP_BCAST,
  REAL,
  CONJ,
  DUMMY,
  ENABLE_PMU,
  DISABLE_PMU,
  ADAM,
  ADAM_BACKWARD,
  DROPOUT,
  LORA_MATMUL_FORWARD,
  OP_NUM,
  COMPARISON_C,
  PPL_EXECUTE
}
OpType;

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
  unsigned count_[OP_NUM];
  bool is_paused_ = false;
  bool is_start_ = false;
  std::mutex mutex_;
  static OpTimer * instance_;
};

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

#ifdef TPU_OP_TIMING
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

#endif // TPU_OP_TIMING

void print_python_code();

} // namespace tpu
