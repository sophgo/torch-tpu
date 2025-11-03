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

#include "TPUDeviceManager.h"
#include "TPUGuard.h"
#include <tpuDNN.h>
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"
#include "TPUStream.h"
#include "OpTimer.h"

#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, func)    \
    AT_DISPATCH_SWITCH(                                         \
        scalar_type, name,                                      \
        AT_DISPATCH_CASE(at::kFloat, func)                      \
        AT_DISPATCH_CASE(at::kHalf, func)                       \
        AT_DISPATCH_CASE(at::kBFloat16, func)                   \
        AT_DISPATCH_CASE(at::kInt, func)                        \
        AT_DISPATCH_CASE(at::kShort, func)                      \
        AT_DISPATCH_CASE(at::kChar, func)                       \
        AT_DISPATCH_CASE(at::kByte, func))

static inline bool usePPLKernels()
{
    auto flagStr = std::getenv("USE_PPL");
    if (!flagStr)
    {
#ifdef BACKEND_SG2260E
        return true;
#else
        return false;
#endif
    }

    return std::atoi(flagStr);
}

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

#ifdef SHOW_OP_INFO
#define SHOW_TENSOR_OP(...) DEBUG_SHOW(at::Tensor, tpu::GetTensorInfo, __VA_ARGS__)
#define SHOW_EMPTY_INFO(Tensor) std::cout << __func__ << ": " << Tensor.data_ptr() << std::endl;
#else
#define SHOW_TENSOR_OP(...)
#define SHOW_EMPTY_INFO(Tensor)
#endif

#ifdef DEBUG
#define CPU_IMPL_WARNING(...)   LOG( WARNING ) << __func__ << " use cpu impl." << #__VA_ARGS__;
#define CONTIGUOUS_WARNING(...) LOG( WARNING ) << __func__ << " will contiguous." << #__VA_ARGS__;

#define CHECK_TENSOR_IN_DEVICE(t) \
do \
{ \
TORCH_CHECK ( t.device().type() == DeviceType::TPU, ":", __func__, ": ", #t, " is not in TPU device" ); \
TORCH_CHECK ( t.is_contiguous() == true, __FILE__ ,":" ,__func__, ": ", #t, " is not contiguous" ); \
SHOW_TENSOR_OP(t); \
} \
while ( 0 )

#define CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(t) \
do \
{ \
TORCH_CHECK ( t.device().type() == DeviceType::TPU, #t, " is not in TPU device" ); \
} \
while ( 0 )
#else

#define CPU_IMPL_WARNING(...)
#define CONTIGUOUS_WARNING(...)
#define CHECK_TENSOR_IN_DEVICE(t) (void)(t)
#define CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(t) (void)(t)

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

static inline tpudnnTensor_t TPUGenerateTpudnnTensor(tpudnnHandle_t handle, const at::Tensor & Tensor)
{
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(Tensor);

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

static inline tpudnnTensor_t TPUGenerateTpudnnTensorforComplex64(tpudnnHandle_t handle, const at::Tensor & tensor)
{
  auto ret = TPUGenerateTensorforComplex64<tpudnnTensor_t>(tensor);
  ret.addr = tpudnnPhysToVirt(handle, (unsigned long long)ret.addr);
  return ret;
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

class OpCalDtype;

} // namespace tpu
