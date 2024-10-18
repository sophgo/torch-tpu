

#include "sg_api_struct.h"
#include "sgdnn_api.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>

#if defined BACKEND_1684X
#include "torch_tpu_kernel_data.h"
#elif defined BACKEND_SG2260
#include "tpukernel_launcher.hpp"
#endif

#ifdef USING_PERF_MODE
#include <iostream>
#endif

#define SGDNN_UNUSED(x) (void)(x)

static void sgdnn_show_tensor_info(SgdnnTensor_t tensor)
{
  auto dim = tensor.dim;
  auto shape = tensor.shape;
  auto dtype = tensor.dtype;
  printf(">>>> Tensor Info: dim: %d, dtype: %d, shape: [", dim, dtype);
  for (int i = 0; i < dim; i++)
  {
    printf("%d, ", shape[i]);
  }
  printf("]\n");
}

static inline size_t sgdnnDataSize ( SgdnnDataType_t dtype )
{
  if ( dtype == SGDNN_DTYPE_INT8 ||
       dtype == SGDNN_DTYPE_UINT8 )
  {
    return 1;
  }
  else if ( dtype == SGDNN_DTYPE_INT16 ||
            dtype == SGDNN_DTYPE_UINT16 ||
            dtype == SGDNN_DTYPE_FP16 ||
            dtype == SGDNN_DTYPE_BF16 )
  {
    return 2;
  }
  else if ( dtype == SGDNN_DTYPE_FP32 ||
            dtype == SGDNN_DTYPE_INT32 ||
            dtype == SGDNN_DTYPE_UINT32 )
  {
    return 4;
  }
  else if ( dtype == SGDNN_DTYPE_INT64 )
  {
    return 8;
  }
  return -1;
}

static inline bool sgdnnIsTensorContiguous ( const SgdnnTensor_t * tensor )
{
  int stride = 1;
  for ( int i = tensor->dim - 1; i >= 0; --i )
  {
    if ( tensor->shape[i] > 1 && tensor->stride[i] != stride )
    {
      return false;
    }
    else
    {
      stride *= tensor->shape[i];
    }
  }
  return true;
}

static inline bool sgdnnIsTensorTransposed ( const SgdnnTensor_t * tensor )
{
  if ( tensor->dim < 2 || sgdnnIsTensorContiguous ( tensor ) )
  {
    return false;
  }
  else
  {
    int stride = 1;
    for ( int i = tensor->dim - 1; i >= 0; --i )
    {
      if ( ( i == tensor->dim - 1 && tensor->stride[i] != tensor->shape[tensor->dim - 2] ) ||
           ( i == tensor->dim - 2 && tensor->stride[i] != 1 ) ||
           ( i < tensor->dim - 2 && tensor->stride[i] != stride ) )
      {
        return false;
      }
      else
      {
        stride *= tensor->shape[i];
      }
    }
  }
  return true;
}

static inline bool sgdnnIsSameShape ( const SgdnnTensor_t * tensor1, const SgdnnTensor_t * tensor2 )
{
  if ( tensor1->dim == tensor2->dim )
  {
    for ( int i = 0; i < tensor1->dim; ++i )
    {
      if ( tensor1->shape[i] != tensor2->shape[i] )
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

void sgdnn32ICShape ( const int * shape, int * _32ic_shape )
{
  _32ic_shape[0] = shape[0];
  _32ic_shape[1] = shape[2] * shape[3];
  _32ic_shape[2] = DIV_UP ( shape[1], 32 );
  _32ic_shape[3] = 32;
}

void sgdnn32OCShape ( const int * shape, int * _32oc_shape )
{
  _32oc_shape[0] = shape[1];
  _32oc_shape[1] = shape[2] * shape[3];
  _32oc_shape[2] = DIV_UP ( shape[0], 32 );
  _32oc_shape[3] = 32;
}

void sgdnnContiguousStride ( const int * shape, int dim,  int * stride )
{
  int s = 1;
  for ( int i = dim - 1; i >= 0; --i )
  {
    stride[i] = s;
    s *= shape[i];
  }
}

static inline size_t sgdnnTensorBytes ( const SgdnnTensor_t * tensor )
{
  size_t bytes = sgdnnDataSize ( tensor->dtype );
  for ( int i = 0; i < tensor->dim; ++i )
  {
    bytes *= tensor->shape[i];
  }
  return bytes;
}

static inline bool sgdnnDtypeIsInt64 (SgdnnDataType_t dtype)
{
  return dtype == SGDNN_DTYPE_INT64;
}

static inline int sgdnnTPUKernelDType ( SgdnnDataType_t dtype )
{
#if defined BACKEND_1684X
  if ( dtype == SGDNN_DTYPE_INT8 )        { return ( 0 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT8 )  { return ( 0 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_INT16 )  { return ( 3 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT16 ) { return ( 3 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_FP16 )   { return ( 1 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_BF16 )   { return ( 5 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_INT32 )  { return ( 4 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT32 ) { return ( 4 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_FP32 )   { return ( 2 << 1 ) | 1; }
#elif defined BACKEND_SG2260
  if ( dtype == SGDNN_DTYPE_INT8 )        { return ( 0 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT8 )  { return ( 0 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_INT16 )  { return ( 3 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT16 ) { return ( 3 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_FP16 )   { return ( 1 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_BF16 )   { return ( 5 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_INT32 )  { return ( 4 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT32 ) { return ( 4 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_FP32 )   { return ( 2 << 1 ) | 1; }
#else
  SGDNN_CHECK ( false );
#endif
  printf ( "Data type %d is not supported by the current backend\n", dtype );
  SGDNN_CHECK ( false );
}

tpu_status_t sgdnnInitialize( tpu_resource_t resource );

#if defined BACKEND_1684X
static std::map<tpu_resource_t, tpu_kernel_module_t> tpu_kernel_module;
tpu_kernel_module_t get_kernel_module(bm_handle_t handle)
{
  sgdnnInitialize(handle);
  return tpu_kernel_module[handle];
}
#elif defined BACKEND_SG2260
static TPUKernelLauncher kernel_launcher;
tpuRtKernelModule_t get_kernel_module(tpuRtStream_t stream)
{
  return kernel_launcher.get_kernel_module(stream);
}
#endif

tpu_status_t sgdnnInitialize( tpu_resource_t resource )
{
#if defined BACKEND_1684X
  if ( tpu_kernel_module.find ( resource  ) != tpu_kernel_module.end() )
  {
    return SG_SUCCESS;
  }
  const char* p = torch_tpu_kernel_data;
  const size_t length = torch_tpu_kernel_data_length;
  tpu_kernel_module_t tpu_module = tpu_kernel_load_module ( resource , ( const char * ) p, length );
  tpu_kernel_module.insert ( std::pair<tpu_resource_t, tpu_kernel_module_t> ( resource , tpu_module ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK( kernel_launcher.register_kernel_module(resource) == SG_SUCCESS);
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnDeinitialize ( tpu_resource_t resource  )
{
#if defined BACKEND_1684X
  if ( tpu_kernel_module.find ( resource  ) == tpu_kernel_module.end() )
  {
    return SG_SUCCESS;
  }
  SGDNN_CHECK ( tpu_kernel_module.erase ( resource  ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( kernel_launcher.unload_kernel_module(resource) == SG_SUCCESS );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnTPUKernelLaunch (
            tpu_resource_t resource,
            const char * func_name,
            const void * api,
            size_t api_size,
            bool non_blocking,
            int group_num,
            int block_num )
{
#if defined BACKEND_1684X
  tpu_kernel_function_t func_id;
  tpu_kernel_module_t tpu_module = tpu_kernel_module[resource ];
  func_id = tpu_kernel_get_function ( resource , tpu_module, func_name );
  return tpu_kernel_launch ( resource , func_id, ( void * ) api, api_size );
#elif defined BACKEND_SG2260
  const char* core_num_env = getenv("TPUTRAIN_CORE_NUM");
  if (core_num_env) {
    group_num = 1;
    block_num = atoi(core_num_env);
  }
  if (non_blocking)
    return kernel_launcher.launch_async( func_name, api, api_size, resource, group_num, block_num );
  else
    return kernel_launcher.launch_sync( func_name, api, api_size, resource, group_num, block_num );
#else
  SGDNN_CHECK ( false );
#endif
}

#if defined BACKEND_SG2260
tpu_status_t sgdnnCacheMalloc(void** dev_ptr, int64_t size)
{
  return kernel_launcher.cache_malloc(dev_ptr, size);
}

tpu_status_t sgdnnCacheFree( void* dev_ptr, tpu_resource_t resource )
{
  return kernel_launcher.cache_free(dev_ptr, resource);
}
#endif

tpu_status_t sgdnnStridedCopy (   tpu_resource_t  resource ,
                               SgdnnTensor_t input,
                               SgdnnTensor_t output,
                               bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype != SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  sg_api_strided_copy_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
    api.input_stride[i] = input.stride[i];
    api.output_stride[i] = output.stride[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource, "tpu_kernel_api_strided_copy", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_strided_copy_multi_core", &api, sizeof ( api ), non_blocking);
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnDummy ( tpu_resource_t  resource ,
                          bool non_blocking )
{
  sg_api_gelu_t api = {0}; // no use
  sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_dummy", &api, sizeof ( api ), non_blocking );
  return SG_SUCCESS;
}

tpu_status_t sgdnnDummy_WO_KERNEL_LAUNCH ( tpu_resource_t  resource ,
                           bool non_blocking )
{
  return SG_SUCCESS;
}

tpu_status_t sgdnnFormatCast( tpu_resource_t  resource,
                              SgdnnTensor_t input,
                              SgdnnTensor_t output,
                              int cast_type)
{
  if (cast_type == 0) {// 0 : conv weight forward 32 ic, and dtype is fp16. infer mode.
    sgdnnReorderConv2dWeight(resource, input, 0, output);
  } else if (cast_type == 1) { // 1: origin conv weight storage with 32IC, Cast to Contiguous. the inversion of case 0.
    sgdnnRecoverConv2dWeight(resource, input, 0, output);
  } else if (cast_type == 2) { // 2: origin conv weight is Contiguous. Cast to 32IC32OC(double Memory). Train mode.
    sgdnnReorderConv2dWeight(resource, input, 2, output);
  } else if (cast_type == 3) { // 3: origin conv weight storage with 32IC32OC, Cast to Contiguous. the inversion of case 2.
    sgdnnRecoverConv2dWeight(resource, input, 2, output);
  } else if (cast_type == 4) { // 4:  conv fp16 32OC weight'grad to contiguous format(ND)
    sgdnnRecoverConv2dGrad( resource, input, output);
  } else if (cast_type == 5) { // 5: conv  contiguous format(ND) weight'grad tofp16 32OC
    sgdnnReorderConv2dGrad( resource, input, output);
  }
  else {
    // TODO: support other case
    SGDNN_CHECK ( false );
  }
  return SG_SUCCESS;
}

tpu_status_t sgdnnConvertInt64toInt32 ( tpu_resource_t resource ,
                                      SgdnnTensor_t input,
                                      SgdnnTensor_t output,
                                      bool non_blocking)
{
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  SgdnnTensor_t input_ = input;
  input_.dtype = SGDNN_DTYPE_INT32;
  for (int i = input.dim - 1; i >=0; i--)
  {
    input_.stride[i] = input.stride[i] * 2;
  }
  tpu_status_t status = sgdnnStridedCopy(resource , input_, output, non_blocking);
  return status;
}


tpu_status_t sgdnnConv2d ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight,
                          SgdnnTensor_t bias,
                          SgdnnConv2dParam_t param,
                          SgdnnTensor_t output,
                          bool non_blocking)
{
  SGDNN_CHECK ( input.dtype == weight.dtype );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == bias.dtype );
  }
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input.dim == 4 );
  SGDNN_CHECK ( weight.dim == 4 || weight.dim == 8 );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.dim == 1 );
  }
  SGDNN_CHECK ( output.dim == 4 );
  SGDNN_CHECK ( input.shape[0] == output.shape[0] );
  SGDNN_CHECK ( input.shape[1] % param.groups == 0 );
  SGDNN_CHECK ( output.shape[1] % param.groups == 0 );
  SGDNN_CHECK ( weight.shape[0] = output.shape[1] );
  SGDNN_CHECK ( weight.shape[1] = input.shape[1] / param.groups );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.shape[0] == output.shape[1] );
  }
  // TODO: CHECK H, W
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  if ( !weight.format_casted ) {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &bias ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  tpu_device_mem_t weight_reordered_mem, bias_fp32_mem;
  size_t weight_reordered_bytes = 0;
  if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
  {
    SgdnnTensor_t weight_reordered, bias_f32;
    if ( !weight.format_casted ) {  // no format cast when init, cast in runtime
      weight_reordered.dim = 4;
      weight_reordered.dtype = weight.dtype;
      sgdnn32ICShape ( weight.shape, weight_reordered.shape );
      weight_reordered_bytes = sgdnnTensorBytes ( &weight_reordered );
      sgdnnContiguousStride ( weight_reordered.shape, 4, weight_reordered.stride );
      weight_reordered.dtype = weight.dtype;
      SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &weight_reordered_mem, sgdnnTensorBytes ( &weight_reordered ) ) );
      weight_reordered.addr = sgdnnGetDeviceAddr ( weight_reordered_mem );
      SAFE_CALL ( sgdnnReorderConv2dWeight ( resource , weight, 0, weight_reordered, non_blocking ) );
    }
    if (bias.addr != 0){
      bias_f32 = bias;
      bias_f32.dtype = SGDNN_DTYPE_FP32;
      SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &bias_fp32_mem, sgdnnTensorBytes ( &bias_f32 ) ) );
      bias_f32.addr = sgdnnGetDeviceAddr ( bias_fp32_mem );
      SAFE_CALL ( sgdnnConvert ( resource , bias, bias_f32, non_blocking ) );
    }
  }
  sg_api_conv2d_t api;
  api.input_global_addr = input.addr;
  if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
  {
    if (!weight.format_casted){
      api.weight_global_addr = sgdnnGetDeviceAddr ( weight_reordered_mem );
      api.bias_global_addr = bias.addr != 0 ? sgdnnGetDeviceAddr ( bias_fp32_mem ) : 0 ;
    }
    else {
      api.weight_global_addr = weight.addr;
      api.bias_global_addr = bias.addr != 0 ? sgdnnGetDeviceAddr ( bias_fp32_mem ) : 0 ;
    }
  }
  else
  {
    api.weight_global_addr = weight.addr;
    api.bias_global_addr = bias.addr;
  }
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.groups = param.groups;
  api.output_c = output.shape[1];
  api.kernel[0] = param.kernel_h;
  api.kernel[1] = param.kernel_w;
  api.stride[0] = param.stride_h;
  api.stride[1] = param.stride_w;
  api.dilation[0] = param.dilation_h;
  api.dilation[1] = param.dilation_w;
  api.pad[0] = param.pad_h;
  api.pad[1] = param.pad_h;
  api.pad[2] = param.pad_w;
  api.pad[3] = param.pad_w;
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv2d", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_conv2d_multi_core", &api, sizeof ( api ), non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
  {
    if (!weight.format_casted) {
      sgdnnFreeDevice ( resource , weight_reordered_mem );
    }
    if (bias.addr != 0){
      sgdnnFreeDevice ( resource , bias_fp32_mem );
    }
  }
  SGDNN_UNUSED ( weight_reordered_bytes );
  return SG_SUCCESS;
}

tpu_status_t sgdnnConv2dBackward ( tpu_resource_t resource ,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t weight,
                                  SgdnnConv2dParam_t param,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_weight,
                                  SgdnnTensor_t grad_bias,
                                  bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == weight.dtype );
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_input.dtype );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_weight.dtype );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_bias.dtype );
  }
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input.dim == 4 );
  SGDNN_CHECK ( weight.dim == 4 || weight.dim == 8);
  SGDNN_CHECK ( grad_output.dim == 4 );
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( grad_input.dim == 4 );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( grad_weight.dim == 4 );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( grad_bias.dim == 1 );
  }
  SGDNN_CHECK ( input.shape[0] == grad_output.shape[0] );
  SGDNN_CHECK ( input.shape[1] % param.groups == 0 );
  SGDNN_CHECK ( grad_output.shape[1] % param.groups == 0 );
  //SGDNN_CHECK ( weight.shape[0] = grad_output.shape[1] );
  //SGDNN_CHECK ( weight.shape[1] = input.shape[1] / param.groups );
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( grad_bias.shape[0] == grad_output.shape[1] );
  }
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  }
  if ( grad_weight.addr != 0 )
  {
    // SGDNN_CHECK ( sgdnnIsSameShape ( &weight, &grad_weight ) );
  }
  // TODO: CHECK SHAPE
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  //SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_weight ) );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_bias ) );
  }

  if ( grad_input.addr != 0 || grad_weight.addr != 0 || grad_bias.addr != 0 )
  {
    tpu_device_mem_t buffer_mem;
    if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
    {
      size_t weight_reordered_size;
      if ( weight.format_casted )
      {
        weight_reordered_size = 0;
      }
      else {
        SgdnnTensor_t weight_reordered;
        weight_reordered.dim = 4;
        sgdnn32OCShape ( weight.shape, weight_reordered.shape );
        weight_reordered.dtype = weight.dtype;
        weight_reordered_size = sgdnnTensorBytes ( &weight_reordered );
      }

      SgdnnTensor_t grad_output_reordered;
      grad_output_reordered.dim = 4;
      sgdnn32OCShape ( grad_output.shape, grad_output_reordered.shape );
      grad_output_reordered.dtype = grad_output.dtype;
      size_t grad_output_reordered_size = sgdnnTensorBytes ( &grad_output_reordered );
      size_t buffer_size = weight_reordered_size > grad_output_reordered_size ? weight_reordered_size : grad_output_reordered_size;
      SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &buffer_mem, buffer_size ) );
    }
    sg_api_conv2d_backward_t api;
    api.input_global_addr = input.addr;
    api.weight_global_addr = weight.addr;
    api.grad_output_global_addr = grad_output.addr;
    api.grad_input_global_addr = grad_input.addr;
    api.grad_weight_global_addr = grad_weight.addr;
    api.grad_bias_global_addr = grad_bias.addr;
    api.buffer_global_addr = weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 ?
                             sgdnnGetDeviceAddr ( buffer_mem ) : 0;
    for ( int i = 0; i < input.dim; ++i )
    {
      api.input_shape[i] = input.shape[i];
    }
    for ( int i = 0; i < grad_output.dim; ++i )
    {
      api.output_shape[i] = grad_output.shape[i];
    }
    api.dtype = sgdnnTPUKernelDType ( input.dtype );
    api.groups = param.groups;
    api.kernel[0] = param.kernel_h;
    api.kernel[1] = param.kernel_w;
    api.stride[0] = param.stride_h;
    api.stride[1] = param.stride_w;
    api.dilation[0] = param.dilation_h;
    api.dilation[1] = param.dilation_w;
    api.pad[0] = param.pad_h;
    api.pad[1] = param.pad_h;
    api.pad[2] = param.pad_w;
    api.pad[3] = param.pad_w;
    api.weight_formated = weight.format_casted;
    #if defined BACKEND_1684X
    SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv2d_backward", &api, sizeof ( api ) ) );
    #elif defined BACKEND_SG2260
    SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_conv2d_backward_multi_core", &api, sizeof ( api ), non_blocking) );
    #else
    SGDNN_CHECK ( false );
    #endif
    if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
    {
      sgdnnFreeDevice ( resource , buffer_mem );
    }
  }
  return SG_SUCCESS;
}

tpu_status_t sgdnnBatchnorm2d ( tpu_resource_t resource ,
                               SgdnnTensor_t input,
                               SgdnnTensor_t weight,
                               SgdnnTensor_t bias,
                               float eps,
                               SgdnnTensor_t running_mean,
                               SgdnnTensor_t running_var,
                               float momentum,
                               SgdnnTensor_t output,
                               SgdnnTensor_t saved_mean,
                               SgdnnTensor_t saved_invstd,
                                bool non_blocking )
{
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == weight.dtype );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == bias.dtype );
  }
  if ( running_mean.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == running_mean.dtype );
  }
  if ( running_var.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == running_var.dtype );
  }
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == saved_mean.dtype );
  SGDNN_CHECK ( input.dtype == saved_invstd.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input.dim == 3 || input.dim == 4 );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( weight.dim == 1 );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.dim == 1 );
  }
  if ( running_mean.addr != 0 )
  {
    SGDNN_CHECK ( running_mean.dim == 1 );
  }
  if ( running_var.addr != 0 )
  {
    SGDNN_CHECK ( running_var.dim == 1 );
  }
  SGDNN_CHECK ( output.dim == input.dim );
  SGDNN_CHECK ( saved_mean.dim == 1 );
  SGDNN_CHECK ( saved_invstd.dim == 1 );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( weight.shape[0] == input.shape[1] );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.shape[0] == input.shape[1] );
  }
  if ( running_mean.addr != 0 )
  {
    SGDNN_CHECK ( running_mean.shape[0] == input.shape[1] );
  }
  if ( running_var.addr != 0 )
  {
    SGDNN_CHECK ( running_var.shape[0] == input.shape[1] );
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( saved_mean.shape[0] == input.shape[1] );
  SGDNN_CHECK ( saved_invstd.shape[0] == input.shape[1] );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &bias ) );
  }
  if ( running_mean.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &running_mean ) );
  }
  if ( running_var.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &running_var ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &saved_mean ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &saved_invstd ) );

  sg_api_batchnorm2d_t api;
  api.input_global_addr = input.addr;
  api.running_mean_global_addr = running_mean.addr;
  api.running_var_global_addr = running_var.addr;
  api.weight_global_addr = weight.addr;
  api.bias_global_addr = bias.addr;
  api.saved_mean_global_addr = saved_mean.addr;
  api.saved_invstd_global_addr = saved_invstd.addr;
  api.output_global_addr = output.addr;
  api.momentum = momentum;
  api.eps = eps;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  if ( input.dim == 3 )
  {
    api.shape[3] = 1;
  }
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_batchnorm2d", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL(  sgdnnTPUKernelLaunchMultiCore( resource , "tpu_kernel_api_batchnorm2d_multi_core", &api, sizeof ( api ) , non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnBatchnorm2dBackward ( tpu_resource_t resource ,
                                       SgdnnTensor_t grad_output,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t weight,
                                       SgdnnTensor_t saved_mean,
                                       SgdnnTensor_t saved_invstd,
                                       SgdnnTensor_t grad_input,
                                       SgdnnTensor_t grad_weight,
                                       SgdnnTensor_t grad_bias,
                                      bool non_blocking )
{
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == weight.dtype );
  }
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_input.dtype );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_weight.dtype );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_bias.dtype );
  }
  SGDNN_CHECK ( input.dtype == saved_mean.dtype );
  SGDNN_CHECK ( input.dtype == saved_invstd.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input.dim == 3 || input.dim == 4 );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( weight.dim == 1 );
  }
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( grad_input.dim == input.dim );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( grad_weight.dim == 1 );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( grad_bias.dim == 1 );
  }
  SGDNN_CHECK ( grad_output.dim == input.dim );
  SGDNN_CHECK ( saved_mean.dim == 1 );
  SGDNN_CHECK ( saved_invstd.dim == 1 );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( weight.shape[0] == input.shape[1] );
  }
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( grad_weight.shape[0] == input.shape[1] );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( grad_bias.shape[0] == input.shape[1] );
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_output ) );
  SGDNN_CHECK ( saved_mean.shape[0] == input.shape[1] );
  SGDNN_CHECK ( saved_invstd.shape[0] == input.shape[1] );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  }
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_weight ) );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_bias ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &saved_mean ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &saved_invstd ) );
#if defined BACKEND_1684X
  sg_api_batchnorm2d_backward_t api;
  api.grad_output_global_addr = grad_output.addr;
  api.input_global_addr = input.addr;
  api.weight_global_addr = weight.addr;
  api.saved_mean_global_addr = saved_mean.addr;
  api.saved_invstd_global_addr = saved_invstd.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.grad_weight_global_addr = grad_weight.addr;
  api.grad_bias_global_addr = grad_bias.addr;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  if ( input.dim == 3 )
  {
    api.shape[3] = 1;
  }
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_batchnorm2d_backward", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLayernorm ( tpu_resource_t resource ,
                             SgdnnTensor_t input,
                             SgdnnTensor_t weight,
                             SgdnnTensor_t bias,
                             int start_dim,
                             float eps,
                             SgdnnTensor_t output,
                             SgdnnTensor_t mean,
                             SgdnnTensor_t rstd,
                             bool non_blocking )
{
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == weight.dtype );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == bias.dtype );
  }
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == mean.dtype );
  SGDNN_CHECK ( input.dtype == rstd.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( start_dim < 0 )
  {
    start_dim += input.dim;
  }
  SGDNN_CHECK ( start_dim > 0 && start_dim < input.dim );
  SGDNN_CHECK ( input.dim >= 2 );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( weight.dim == input.dim - start_dim );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.dim == input.dim - start_dim );
  }
  SGDNN_CHECK ( output.dim == input.dim );
  SGDNN_CHECK ( mean.dim == input.dim );
  SGDNN_CHECK ( rstd.dim == input.dim );
  if ( weight.addr != 0 )
  {
    for ( int i = start_dim; i < input.dim; ++i )
    {
      SGDNN_CHECK ( weight.shape[i - start_dim] == input.shape[start_dim] );
    }
  }
  if ( bias.addr != 0 )
  {
    for ( int i = start_dim; i < input.dim; ++i )
    {
      SGDNN_CHECK ( bias.shape[i - start_dim] == input.shape[start_dim] );
    }
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  for ( int i = 0; i < input.dim; ++i )
  {
    if ( i < start_dim )
    {
      SGDNN_CHECK ( mean.shape[i] == input.shape[i] );
    }
    else
    {
      SGDNN_CHECK ( mean.shape[i] == 1 );
    }
  }
  for ( int i = 0; i < input.dim; ++i )
  {
    if ( i < start_dim )
    {
      SGDNN_CHECK ( rstd.shape[i] == input.shape[i] );
    }
    else
    {
      SGDNN_CHECK ( rstd.shape[i] == 1 );
    }
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  }
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &bias ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &mean ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &rstd ) );
  sg_api_layernorm_t api;
  api.input_global_addr = input.addr;
  api.weight_global_addr = weight.addr;
  api.bias_global_addr = bias.addr;
  api.mean_global_addr = mean.addr;
  api.rstd_global_addr = rstd.addr;
  api.output_global_addr = output.addr;
  api.eps = eps;
  api.axis = start_dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dim = input.dim;
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_layernorm", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_layernorm_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLayernormBackward ( tpu_resource_t resource ,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t input,
                                     SgdnnTensor_t weight,
                                     SgdnnTensor_t mean,
                                     SgdnnTensor_t rstd,
                                     int start_dim,
                                     SgdnnTensor_t grad_input,
                                     SgdnnTensor_t grad_weight,
                                     SgdnnTensor_t grad_bias,
                                     int requires_grad_input,
                                     bool non_blocking )
{
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == weight.dtype );
  }
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  SGDNN_CHECK ( input.dtype == mean.dtype );
  SGDNN_CHECK ( input.dtype == rstd.dtype );
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_input.dtype );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_weight.dtype );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_bias.dtype );
  }
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( start_dim < 0 )
  {
    start_dim += input.dim;
  }
  SGDNN_CHECK ( start_dim > 0 && start_dim < input.dim );
  SGDNN_CHECK ( input.dim >= 2 );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( weight.dim == input.dim - start_dim );
  }
  SGDNN_CHECK ( mean.dim == input.dim );
  SGDNN_CHECK ( rstd.dim == input.dim );
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( grad_bias.dim == input.dim - start_dim );
  }
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsSameShape ( &weight, &grad_weight ) );
  }
  if ( weight.addr != 0 )
  {
    for ( int i = start_dim; i < input.dim; ++i )
    {
      SGDNN_CHECK ( weight.shape[i - start_dim] == input.shape[start_dim] );
    }
  }
  if ( grad_bias.addr != 0 )
  {
    for ( int i = start_dim; i < input.dim; ++i )
    {
      SGDNN_CHECK ( grad_bias.shape[i - start_dim] == input.shape[start_dim] );
    }
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_output ) );
  for ( int i = 0; i < input.dim; ++i )
  {
    if ( i < start_dim )
    {
      SGDNN_CHECK ( mean.shape[i] == input.shape[i] );
    }
    else
    {
      SGDNN_CHECK ( mean.shape[i] == 1 );
    }
  }
  for ( int i = 0; i < input.dim; ++i )
  {
    if ( i < start_dim )
    {
      SGDNN_CHECK ( rstd.shape[i] == input.shape[i] );
    }
    else
    {
      SGDNN_CHECK ( rstd.shape[i] == 1 );
    }
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  if ( weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &mean ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &rstd ) );
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  }
  if ( grad_weight.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_weight ) );
  }
  if ( grad_bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_bias ) );
  }
  sg_api_layernorm_backward_t api;
  api.grad_output_global_addr = grad_output.addr;
  api.input_global_addr = input.addr;
  api.weight_global_addr = weight.addr;
  api.mean_global_addr = mean.addr;
  api.rstd_global_addr = rstd.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.grad_weight_global_addr = grad_weight.addr;
  api.grad_bias_global_addr = grad_bias.addr;
  api.axis = start_dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dim = input.dim;
  api.requires_grad_input = requires_grad_input;
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << 5 <<std::endl;
  std::cout << "output_num\n" << 3 <<std::endl;
  std::cout << "input_addr0\n" << api.grad_output_global_addr << std::endl;
  std::cout << "input_size0\n" << sgdnnTensorBytes(&grad_output) << std::endl;
  std::cout << "input_addr1\n" << api.input_global_addr << std::endl;
  std::cout << "input_size1\n" << sgdnnTensorBytes(&input) << std::endl;
  std::cout << "input_addr2\n" << api.weight_global_addr << std::endl;
  std::cout << "input_size2\n" << sgdnnTensorBytes(&weight) << std::endl;
  std::cout << "input_addr3\n" << api.mean_global_addr << std::endl;
  std::cout << "input_size3\n" << sgdnnTensorBytes(&mean) << std::endl;
  std::cout << "input_addr4\n" << api.rstd_global_addr << std::endl;
  std::cout << "input_size4\n" << sgdnnTensorBytes(&rstd) << std::endl;
  std::cout << "output_addr0\n" << api.grad_input_global_addr << std::endl;
  std::cout << "output_size0\n" << sgdnnTensorBytes(&grad_input) << std::endl;
  std::cout << "output_addr1\n" << api.grad_weight_global_addr << std::endl;
  std::cout << "output_size1\n" << sgdnnTensorBytes(&grad_weight) << std::endl;
  std::cout << "output_addr2\n" << api.grad_bias_global_addr << std::endl;
  std::cout << "output_size2\n" << sgdnnTensorBytes(&grad_bias) << std::endl;
#endif
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_layernorm_backward", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_layernorm_backward_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnReorderConv2dWeight ( tpu_resource_t resource ,
                                       SgdnnTensor_t input,
                                       int mode,
                                       SgdnnTensor_t output,
                                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( mode == 0 || mode == 1 || mode == 2);
  if ( mode == 0 ) // 32 IC
  {
    SGDNN_CHECK ( input.dim == 4 );
    SGDNN_CHECK ( output.dim == 4 );
    SGDNN_CHECK ( output.shape[0] == input.shape[0] );
    SGDNN_CHECK ( output.shape[1] == input.shape[2] * input.shape[3] );
    SGDNN_CHECK ( output.shape[2] == DIV_UP ( input.shape[1], 32 ) );
    SGDNN_CHECK ( output.shape[3] == 32 );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  }
  else if ( mode == 1 ) // 32 OC
  {
    SGDNN_CHECK ( input.dim == 4 );
    SGDNN_CHECK ( output.dim == 4 );
    SGDNN_CHECK ( output.shape[0] == input.shape[1] );
    SGDNN_CHECK ( output.shape[1] == input.shape[2] * input.shape[3] );
    SGDNN_CHECK ( output.shape[2] == DIV_UP ( input.shape[0], 32 ) );
    SGDNN_CHECK ( output.shape[3] == 32 );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  }
  else if ( mode == 2 ) // 32IC32OC
  {
    SGDNN_CHECK ( input.dim == 4 );
    SGDNN_CHECK ( output.dim == 8 );
    // 32IC
    SGDNN_CHECK ( output.shape[0] == input.shape[0] );
    SGDNN_CHECK ( output.shape[1] == input.shape[2] * input.shape[3] );
    SGDNN_CHECK ( output.shape[2] == DIV_UP ( input.shape[1], 32 ) );
    SGDNN_CHECK ( output.shape[3] == 32 );
    // 32OC
    SGDNN_CHECK ( output.shape[4] == input.shape[1] );
    SGDNN_CHECK ( output.shape[5] == input.shape[2] * input.shape[3] );
    SGDNN_CHECK ( output.shape[6] == DIV_UP ( input.shape[0], 32 ) );
    SGDNN_CHECK ( output.shape[7] == 32 );
  }
#if defined BACKEND_1684X
  sg_api_conv_weight_reorder_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.mode = mode;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv_weight_reorder", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  // SGDNN_CHECK ( false );
  sg_api_conv_weight_reorder_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.mode = mode;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_conv_weight_reorder_multi_core", &api, sizeof ( api ) , non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

// NCHW -> 32OC
tpu_status_t sgdnnReorderConv2dGrad ( tpu_resource_t resource ,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t output,
                                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( output.dim == 4  && input.dim == 4);

#if defined BACKEND_1684X
  sg_api_conv_grad_reorder_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv_grad_reorder", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnRecoverConv2dGrad ( tpu_resource_t resource ,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t output,
                                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( output.dim == 4  && input.dim == 4);

#if defined BACKEND_1684X
  sg_api_conv_grad_recover_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < output.dim; ++i )
  {
    api.shape[i] = output.shape[i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv_grad_recover", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnRecoverConv2dWeight ( tpu_resource_t resource ,
                                       SgdnnTensor_t input,
                                       int mode,
                                       SgdnnTensor_t output,
                                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input.dim == 4 || input.dim == 8 );
  SGDNN_CHECK ( output.dim == 4 );
  if ( mode == 0 ) // 32 IC -> NCHW
  {
    SGDNN_CHECK ( input.shape[0] == output.shape[0] );
    SGDNN_CHECK ( input.shape[1] == output.shape[2] * output.shape[3] );
    SGDNN_CHECK ( input.shape[2] == DIV_UP ( output.shape[1], 32 ) );
    SGDNN_CHECK ( input.shape[3] == 32 );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  }
  else if ( mode == 2 ) // 32IC32OC -> NCHW
  {
    SGDNN_CHECK ( input.shape[0] == output.shape[0] );
    SGDNN_CHECK ( input.shape[1] == output.shape[2] * output.shape[3] );
    SGDNN_CHECK ( input.shape[2] == DIV_UP ( output.shape[1], 32 ) );
    SGDNN_CHECK ( input.shape[3] == 32 );
    mode = 0;
    // 32OC part - the alternative
    // sg_api_conv_grad_recover_t api;
    // api.input_global_addr = input.addr + (input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]) * 2;
    // api.output_global_addr = output.addr;
    // for ( int i = 0; i < output.dim; ++i )
    // {
    //   api.shape[i] = output.shape[i];
    // }
    // SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv_grad_recover", &api, sizeof ( api ) ) );
    // return SG_SUCCESS;
  }
  else
  {
    SGDNN_CHECK ( false );
  }

#if defined BACKEND_1684X
  sg_api_conv_weight_recover_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < output.dim; ++i )
  {
    api.shape[i] = output.shape[i];
  }
  api.mode = mode;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv_weight_recover", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnGather(tpu_resource_t resource ,
                        SgdnnTensor_t input,
                        SgdnnTensor_t index,
                        SgdnnTensor_t output,
                        int axis,
                        bool non_blocking ) {
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(
    input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
    input.dtype == SGDNN_DTYPE_INT8 || input.dtype == SGDNN_DTYPE_UINT8 ||
    input.dtype == SGDNN_DTYPE_INT16 || input.dtype == SGDNN_DTYPE_UINT16);
  // SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&index));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));

  sg_api_gather_t api;
  api.input_global_addr = input.addr;
  api.index_global_addr = index.addr;
  api.output_global_addr = output.addr;
  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
    api.index_shape[i] = index.shape[i];
  }
  api.dim = input.dim;
  api.axis = axis;
  api.is_index_int64 = index.dtype == SGDNN_DTYPE_INT64;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
#if defined BACKEND_1684X
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_gather", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_gather_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnSoftmax ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           int dim,
                           SgdnnTensor_t output ,
                           bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  if ( dim < 0 )
  {
    dim += input.dim;
  }
  SGDNN_CHECK ( dim >= 0 && dim < input.dim );
  sg_api_softmax_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_softmax", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_softmax_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnSoftmaxBackward ( tpu_resource_t resource ,
                                   SgdnnTensor_t grad_output,
                                   SgdnnTensor_t output,
                                   int dim,
                                   SgdnnTensor_t grad_input,
                                   bool non_blocking )
{
  SGDNN_CHECK ( grad_output.dtype == output.dtype );
  SGDNN_CHECK ( grad_output.dtype == grad_input.dtype );
  SGDNN_CHECK ( grad_output.dtype == SGDNN_DTYPE_FP32 ||
                grad_output.dtype == SGDNN_DTYPE_FP16 ||
                grad_output.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &grad_output, &output ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &grad_output, &grad_input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  if ( dim < 0 )
  {
    dim += grad_output.dim;
  }
  SGDNN_CHECK ( dim >= 0 && dim < grad_output.dim );
  sg_api_softmax_backward_t api;
  api.grad_output_global_addr = grad_output.addr;
  api.output_global_addr = output.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.dim = grad_output.dim;
  for ( int i = 0; i < grad_output.dim; ++i )
  {
    api.shape[i] = grad_output.shape[i];
  }
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( grad_output.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_softmax_backward", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_softmax_backward_multi_core", &api, sizeof ( api ), non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLogSoftmax ( tpu_resource_t resource ,
                            SgdnnTensor_t input,
                            int dim,
                            SgdnnTensor_t output ,
                            bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  if ( dim < 0 )
  {
    dim += input.dim;
  }
  SGDNN_CHECK ( dim >= 0 && dim < input.dim );
  sg_api_log_softmax_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_log_softmax", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_log_softmax_multi_core", &api, sizeof ( api ) , non_blocking ));
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}


tpu_status_t sgdnnClamp (tpu_resource_t resource ,
                        SgdnnTensor_t input,
                        float min,
                        float max,
                        SgdnnTensor_t output,
                        bool non_blocking ){
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  sg_api_clamp_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.min = min;
  api.max = max;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_clamp", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_clamp_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}
tpu_status_t sgdnnReduce ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          int start_dim,
                          int end_dim,
                          int keepdim,
                          int mode,
                          SgdnnTensor_t output ,
                          bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( start_dim < 0 )
  {
    start_dim += input.dim;
  }
  if ( end_dim < 0 )
  {
    end_dim += input.dim;
  }
  if ( end_dim < start_dim )
  {
    int tmp = end_dim;
    end_dim = start_dim;
    start_dim = tmp;
  }
  SGDNN_CHECK ( end_dim > start_dim && end_dim <= input.dim );
  if ( keepdim )
  {
    SGDNN_CHECK ( input.dim == output.dim );
    for ( int i = 0; i < input.dim; ++i )
    {
      if ( i < start_dim )
      {
        SGDNN_CHECK ( input.shape[i] == output.shape[i] );
      }
      else if ( i >= start_dim && i < end_dim )
      {
        SGDNN_CHECK ( output.shape[i] == 1 );
      }
      else if ( i >= end_dim )
      {
        SGDNN_CHECK ( input.shape[i] == output.shape[i] );
      }
    }
  }
  else
  {
    SGDNN_CHECK ( input.dim == output.dim + ( end_dim - start_dim ) );
    for ( int i = 0; i < input.dim; ++i )
    {
      if ( i < start_dim )
      {
        SGDNN_CHECK ( input.shape[i] == output.shape[i] );
      }
      else if ( i >= end_dim )
      {
        SGDNN_CHECK ( input.shape[i] == output.shape[i - ( end_dim - start_dim )] );
      }
    }
  }
#if defined BACKEND_1684X
  SGDNN_CHECK ( start_dim == 0 || end_dim == input.dim );
  sg_api_reduce_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dim = input.dim;
  api.start_dim = start_dim;
  api.end_dim = end_dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.mode = mode;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_reduce", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( start_dim == 0 || end_dim == input.dim );
  sg_api_reduce_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dim = input.dim;
  api.start_dim = start_dim;
  api.end_dim = end_dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.mode = mode;
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << 1 <<std::endl;
  std::cout << "output_num\n" << 1 <<std::endl;
  std::cout << "input_addr0\n" << api.input_global_addr << std::endl;
  std::cout << "input_size0\n" << sgdnnTensorBytes(&input) << std::endl;
  std::cout << "output_addr0\n" << api.output_global_addr << std::endl;
  std::cout << "output_size0\n" << sgdnnTensorBytes(&output) << std::endl;
#endif
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_reduce_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnReduceProd ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              int axis,
                              int keepdim,
                              SgdnnTensor_t output ,
                              bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( axis >= 0 );
  SGDNN_CHECK ( axis < input.dim );

  sg_api_reduce_prod_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  unsigned int size = 1;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
    size *= input.shape[i];
  }
  if(input.dtype == SGDNN_DTYPE_FP32) {
    size *= sizeof(float);
  }
  else {
    size *= sgdnnDataSize(SGDNN_DTYPE_FP16);
  }
  tpu_device_mem_t dev_mem;
  tpu_status_t status = sgdnnMallocDeviceByte(resource , &dev_mem, size);
  if(SG_SUCCESS != status){
    printf("malloc device error \r\n");
    return status;
  }
  api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  api.dim = input.dim;
  api.axis = axis;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_reduce_prod", &api, sizeof ( api ) , non_blocking) );
  return SG_SUCCESS;
}

tpu_status_t sgdnnConvert ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           SgdnnTensor_t output ,
                           bool non_blocking )
{
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_dtype_convert_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.input_dtype = sgdnnTPUKernelDType ( input.dtype );
  api.output_dtype = sgdnnTPUKernelDType ( output.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_dtype_convert", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_dtype_convert_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.input_dtype = sgdnnTPUKernelDType ( input.dtype );
  api.output_dtype = sgdnnTPUKernelDType ( output.dtype );
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << 1 <<std::endl;
  std::cout << "output_num\n" << 1 <<std::endl;
  std::cout << "input_addr0\n" << api.input_global_addr << std::endl;
  std::cout << "input_size0\n" << sgdnnTensorBytes(&input) << std::endl;
  std::cout << "output_addr0\n" << api.output_global_addr << std::endl;
  std::cout << "output_size0\n" << sgdnnTensorBytes(&output) << std::endl;
#endif
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_dtype_convert_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnWhere ( tpu_resource_t resource ,
                         SgdnnTensor_t cond,
                         SgdnnTensor_t self,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking )
{
  SGDNN_CHECK ( self.dtype == output.dtype );
  SGDNN_CHECK ( self.dtype == other.dtype );
  SGDNN_CHECK ( cond.dim == output.dim );
  SGDNN_CHECK ( self.dim == 0 || self.dim == output.dim );
  SGDNN_CHECK ( other.dim == 0 || other.dim == output.dim );
  for ( int i = 0; i < output.dim; ++i )
  {
    SGDNN_CHECK ( cond.shape[i] == 1 || cond.shape[i] == output.shape[i] );
  }
  if ( self.dim > 0 )
  {
    for ( int i = 0; i < output.dim; ++i )
    {
      SGDNN_CHECK ( self.shape[i] == 1 || self.shape[i] == output.shape[i] );
    }
  }
  if ( other.dim > 0 )
  {
    for ( int i = 0; i < output.dim; ++i )
    {
      SGDNN_CHECK ( other.shape[i] == 1 || other.shape[i] == output.shape[i] );
    }
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &cond ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &self ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_where_t api;
  api.cond_global_addr = cond.addr;
  api.self_global_addr = self.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.cond_dtype = sgdnnTPUKernelDType ( cond.dtype );
  api.dtype = sgdnnTPUKernelDType ( output.dtype );
  api.dim = output.dim;
  for ( int i = 0; i < cond.dim; ++i )
  {
    api.cond_shape[i] = cond.shape[i];
  }
  if ( self.dim > 0 )
  {
    for ( int i = 0; i < self.dim; ++i )
    {
      api.self_shape[i] = self.shape[i];
    }
  }
  else
  {
    for ( int i = 0; i < output.dim; ++i )
    {
      api.self_shape[i] = 1;
    }
  }
  if ( other.dim > 0 )
  {
    for ( int i = 0; i < other.dim; ++i )
    {
      api.other_shape[i] = other.shape[i];
    }
  }
  else
  {
    for ( int i = 0; i < output.dim; ++i )
    {
      api.other_shape[i] = 1;
    }
  }
  for ( int i = 0; i < output.dim; ++i )
  {
    api.output_shape[i] = output.shape[i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_where", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_where_multi_core_t api;
  api.output_addr = output.addr;
  api.cond_addr = cond.addr;
  api.self_addr = self.addr;
  api.other_addr = other.addr;
  api.dims = output.dim;
  api.cond_dtype = sgdnnTPUKernelDType ( cond.dtype );
  api.dtype = sgdnnTPUKernelDType ( output.dtype );
  api.self_is_scalar = false;
  api.self_val = 0.f;
  api.other_is_scalar = false;
  api.other_val = 0.f;
  for ( int i = 0; i < cond.dim; ++i )
  {
    api.cond_shape[i] = cond.shape[i];
  }
  if ( self.dim > 0 )
  {
    for ( int i = 0; i < self.dim; ++i )
    {
      api.self_shape[i] = self.shape[i];
    }
  }
  else
  {
    for ( int i = 0; i < output.dim; ++i )
    {
      api.self_shape[i] = 1;
    }
  }
  if ( other.dim > 0 )
  {
    for ( int i = 0; i < other.dim; ++i )
    {
      api.other_shape[i] = other.shape[i];
    }
  }
  else
  {
    for ( int i = 0; i < output.dim; ++i )
    {
      api.other_shape[i] = 1;
    }
  }
  for ( int i = 0; i < output.dim; ++i )
  {
    api.out_shape[i] = output.shape[i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_where_multi_core", &api, sizeof ( api ) , non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnConcat ( tpu_resource_t resource ,
                          const SgdnnTensor_t * inputs,
                          int input_num,
                          int dim,
                          SgdnnTensor_t output ,
                          bool non_blocking )
{
  for ( int i = 0; i < input_num; ++i )
  {
    SGDNN_CHECK ( inputs[i].dtype == output.dtype );
  }
  for ( int i = 0; i < input_num; ++i )
  {
    SGDNN_CHECK ( inputs[i].dim == output.dim );
  }
  if ( dim < 0 )
  {
    dim += output.dim;
  }
  SGDNN_CHECK ( dim >= 0 && dim < output.dim );
  int sum = 0;
  for ( int i = 0; i < input_num; ++i )
  {
    for ( int j = 0; j < output.dim; ++j )
    {
      if ( j != dim )
      {
        SGDNN_CHECK ( inputs[i].shape[j] == output.shape[j] );
      }
      else
      {
        sum += inputs[i].shape[j];
      }
    }
  }
  SGDNN_CHECK ( sum == output.shape[dim] );
  for ( int i = 0; i < input_num; ++i )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &inputs[i] ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  SGDNN_CHECK ( input_num <= FW_MAX_CONCAT_NUM );
  sg_api_concat_t api;
  for ( int i = 0; i < input_num; ++i )
  {
    api.input_global_addrs[i] = inputs[i].addr;
  }
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input_num; ++i )
  {
    for ( int j = 0; j < inputs[i].dim; ++j )
    {
      api.input_shapes[i][j] = inputs[i].shape[j];
    }
  }
  api.input_num = input_num;
  api.dim = output.dim;
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( output.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_concat", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( input_num <= FW_MAX_CONCAT_NUM );
  sg_api_concat_t api;
  for ( int i = 0; i < input_num; ++i )
  {
    api.input_global_addrs[i] = inputs[i].addr;
  }
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input_num; ++i )
  {
    for ( int j = 0; j < inputs[i].dim; ++j )
    {
      api.input_shapes[i][j] = inputs[i].shape[j];
    }
  }
  api.input_num = input_num;
  api.dim = output.dim;
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( output.dtype );
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << input_num <<std::endl;
  std::cout << "output_num\n" << 1 <<std::endl;
  for ( int i = 0; i < input_num; ++i )
  {
    std::cout << "input_addr" << i << "\n" << api.input_global_addrs[i] << std::endl;
    std::cout << "input_size" << i << "\n" << sgdnnTensorBytes(&inputs[i]) << std::endl;
  }
    std::cout << "output_addr0\n" << api.output_global_addr << std::endl;
    std::cout << "output_size0\n" << sgdnnTensorBytes(&output) << std::endl;
#endif
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_concat_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPoolingForward ( tpu_resource_t resource ,
                               SgdnnTensor_t input,
                               SgdnnTensor_t output,
                               PoolingDescriptor_t pooling_desc,
                               bool non_blocking )
{
  //constrainst need to to checked
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_pooling_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
    api.output_shape[i] = output.shape[i];
  }
  api.pooling_desc = pooling_desc;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  if(pooling_desc.mode == POOLING_AVG ){
    SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_avg_pooling", &api, sizeof ( api ) ) );
  }
  // to be implemented
  // else if(pooling_desc.mode == POOLING_MIN ){
  //   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_min_pooling", &api, sizeof ( api ) ) );
  // }
  // else if(pooling_desc.mode == POOLING_MAX){
  //   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_max_pooling", &api, sizeof ( api ) ) );
  //}
  else{
    SGDNN_CHECK ( false );
  }
#elif defined BACKEND_SG2260
  //constrainst need to to checked
  sg_api_pooling_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
    api.output_shape[i] = output.shape[i];
  }
  api.pooling_desc = pooling_desc;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  if(pooling_desc.mode == POOLING_AVG ){
    SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_avg_pooling_multi_core", &api, sizeof ( api ) , non_blocking) );
  }
  // to be implemented
  // else if(pooling_desc.mode == POOLING_MIN ){
  //   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_min_pooling_multi_core", &api, sizeof ( api ) ) );
  // }
  // else if(pooling_desc.mode == POOLING_MAX){
  //   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_max_pooling_multi_core", &api, sizeof ( api ) ) );
  // }
  else{
    SGDNN_CHECK ( false );
  }
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnIndexSelect ( tpu_resource_t resource ,
                               SgdnnTensor_t input,
                               SgdnnTensor_t indices,
                               int dim,
                               SgdnnTensor_t output ,
                               bool non_blocking )
{
  SGDNN_CHECK ( indices.dtype == SGDNN_DTYPE_INT32 ||
                indices.dtype == SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( input.dim + indices.dim - 1 == output.dim );

  for ( int i = 0; i < dim; ++i )
  {
    SGDNN_CHECK ( input.shape[i] == output.shape[i] );
  }
  for ( int i = dim; i < dim + indices.dim; ++i )
  {
    SGDNN_CHECK ( indices.shape[i - dim] == output.shape[i] );
  }
  for ( int i = dim + indices.dim; i < output.dim; ++i )
  {
    SGDNN_CHECK ( input.shape[i - indices.dim + 1] == output.shape[i] );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &indices ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  SgdnnTensor_t indices_ = indices;
  tpu_device_mem_t indices_mem;
  if (indices.dtype == SGDNN_DTYPE_INT64)
  {
    SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &indices_mem, sgdnnTensorBytes ( &indices ) / 2 ) );
    indices_.dtype = SGDNN_DTYPE_INT32;
    indices_.addr = sgdnnGetDeviceAddr ( indices_mem );
    sgdnnConvertInt64toInt32(resource , indices, indices_);
  }

  sg_api_index_select_t api;
  api.input_global_addr = input.addr;
  api.index_global_addr = indices_.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  api.dim = input.dim;
  api.index_num = 1;
  for ( int i = 0; i < indices_.dim; ++i )
  {
    api.index_num *= indices_.shape[i];
  }
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  // tpu_kernel_api_embedding_backward has bugs: warning: when is_index_int64 == true, will overlap index， todo fix it.
  api.is_index_int64 = false;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_index_select", &api, sizeof ( api ) ) );
  if (indices.dtype == SGDNN_DTYPE_INT64){
    sgdnnFreeDevice ( resource , indices_mem );
  }

#elif defined BACKEND_SG2260
  sg_api_index_select_t api;
  api.input_global_addr = input.addr;
  api.index_global_addr = indices.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  api.dim = input.dim;
  api.index_num = 1;
  for ( int i = 0; i < indices.dim; ++i )
  {
    api.index_num *= indices.shape[i];
  }
  api.axis = dim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.is_index_int64 = indices.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_index_select_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMulIndexSelect (tpu_resource_t resource ,
                                SgdnnTensor_t input,
                                SgdnnTensor_t output,
                                std::vector<SgdnnTensor_t>& indices,
                                bool non_blocking )
{
  typedef struct{
    tpu_device_mem_t bm_dev_mem;
    SgdnnTensor_t tensor;
  } Tensor_t;
  tpu_status_t status;
  if ( indices.size() == 1 ){
     status = sgdnnIndexSelect( resource , input, indices[0], 0, output );
    return status;
  }
  else
  {
    Tensor_t cur_input, cur_output;
    cur_input.tensor = input;
    for ( int axis = 0; axis < static_cast<int>(indices.size()); axis++ ){
      // 1. make buffer output
      cur_output.tensor.dim = input.dim;
      cur_output.tensor.dtype = output.dtype;
      for (int i = 0; i < axis; i++ ){
        cur_output.tensor.shape[i] = cur_input.tensor.shape[i];
      }
      cur_output.tensor.shape[axis] = indices[axis].shape[0];
      for (int i = axis + 1; i < input.dim; i++){
        cur_output.tensor.shape[i] = cur_input.tensor.shape[i];
      }
      cur_output.tensor.stride[cur_output.tensor.dim - 1] = 1;
      for (int i = cur_output.tensor.dim - 2; i >=0; i-- )
      {
        cur_output.tensor.stride[i] = cur_output.tensor.shape[i+1] * cur_output.tensor.stride[i+1];
      }
      if (axis == static_cast<int>(indices.size() - 1) )
      {
        cur_output.tensor.addr = output.addr;
      }
      else
      {
        SAFE_CALL( sgdnnMallocDeviceByte ( resource , &cur_output.bm_dev_mem, sgdnnTensorBytes ( &cur_output.tensor ) ) );
        cur_output.tensor.addr = sgdnnGetDeviceAddr(cur_output.bm_dev_mem);
      }
      // 2.cal
      status = sgdnnIndexSelect( resource , cur_input.tensor, indices[axis], axis, cur_output.tensor);
      SGDNN_CHECK(status == SG_SUCCESS);
      // 3. free buffer
      if (axis != 0)
        sgdnnFreeDevice( resource , cur_input.bm_dev_mem );
      cur_input = cur_output;
    }
  }

  return SG_SUCCESS;
}

tpu_status_t sgdnnFill ( tpu_resource_t resource ,
                        const void * scalar_ptr,
                        SgdnnTensor_t output ,
                        bool non_blocking )
{
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_constant_fill_t api;
  api.output_global_addr = output.addr;
  api.dim = output.dim;
  for ( int i = 0; i < output.dim; ++i )
  {
    api.shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( output.dtype );
  api.value = 0;
  for ( size_t i = 0; i < sgdnnDataSize ( output.dtype ); ++i )
  {
    ( ( char * ) &api.value ) [i] = ( ( const char * ) scalar_ptr ) [i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_const_fill", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_constant_fill_t api;
  api.output_global_addr = output.addr;
  api.dim = output.dim;
  for ( int i = 0; i < output.dim; ++i )
  {
    api.shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( output.dtype );
  api.value = 0;
  for ( size_t i = 0; i < sgdnnDataSize ( output.dtype ); ++i )
  {
    ( ( char * ) &api.value ) [i] = ( ( const char * ) scalar_ptr ) [i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_const_fill_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnUpsampling(tpu_resource_t resource , SgdnnTensor_t input,
                            SgdnnTensor_t output,
                            bool align_corners,
                            sg_resize_mode_t upsampling_type,
                            bool non_blocking ) {
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));

  sg_api_upsampling2d_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  tpu_device_mem_t buffer_mem;
  SAFE_CALL( sgdnnMallocDeviceByte ( resource , &buffer_mem, sgdnnTensorBytes ( &output ) ) );
  api.buffer_addr = sgdnnGetDeviceAddr(buffer_mem);
  api.if_getting_buffer_size = false;
  api.platform_sp = upsampling_type == UPSAMPLING_BILINEAR ? PYTORCH_SUPPORT : PYTORCH_NEAREST;

  api.dim = output.dim;
  api.dtype = sgdnnTPUKernelDType(output.dtype);

  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }

  for (int i = 0; i < output.dim; ++i) {
    api.out_shape[i] = output.shape[i];
  }
  api.pad_bag = 0;
  api.pad_end = 0;
  if (upsampling_type == UPSAMPLING_BILINEAR){
    api.half_pixel_centers = 1;
    api.align_corners = align_corners;
  }else{
    api.half_pixel_centers = false;
    api.align_corners = true;
  }
#if defined BACKEND_1684X
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_interp", &api,
                                 sizeof(api)));
  sgdnnFreeDevice(resource , buffer_mem);
#elif defined BACKEND_SG2260
  // SGDNN_CHECK(false);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_interp_multi_core",
                                 &api, sizeof(api), non_blocking));
  sgdnnFreeDevice(resource , buffer_mem);
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnUpsampleNearest2dBackward(tpu_resource_t resource , SgdnnTensor_t grad_output,
                            SgdnnTensor_t grad_input,
                            int scale,
                            PoolingDescriptor_t pooling_desc,
                            bool non_blocking ) {
  SGDNN_CHECK(sgdnnIsTensorContiguous(&grad_output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&grad_input));

  sg_api_upsample2d_backward_t api;
  api.input_global_addr = grad_output.addr;
  api.output_global_addr = grad_input.addr;
  for ( int i = 0; i < grad_output.dim; ++i )
  {
    api.input_shape[i] = grad_output.shape[i];
    api.output_shape[i] = grad_input.shape[i];
  }
  api.pooling_desc = pooling_desc;
  api.scalar = scale;
  api.dtype = sgdnnTPUKernelDType ( grad_output.dtype );
  if(pooling_desc.mode == POOLING_AVG ){
    #if defined BACKEND_1684X
    SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_upsample_nearest2d_backward", &api, sizeof ( api ) ) );
    #elif defined BACKEND_SG2260
    SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_upsample_nearest2d_backward_multi_core", &api, sizeof ( api ) , non_blocking) );
    #else
    SGDNN_CHECK ( false );
    #endif
  } else {
    SGDNN_CHECK ( false );
  }
  return SG_SUCCESS;
}


tpu_status_t sgdnnActive(tpu_resource_t resource , SgdnnTensor_t input,
                        SgdnnTensor_t output, sg_active_type_t active_type,
                        bool non_blocking ) {
  /**
   *
   * ACTIVE_RELU、ACTIVE_ABSVAL
   * 的输入输出可以是FLOAT32/INT8/UINT8/INT16/UINT16，其它数据类型只能是FLOAT32。ACTIVE_RELU、ACTIVE_ABSVAL输入是INT8/UINT8/INT16/UINT16，输出可以是INT8/UINT8/INT16/UINT16，输入输出数据类型不要求一致，结果超出位宽时做饱和处理。
   * BM1684X：ACTIVE_RELU、ACTIVE_ABSVAL、ACTIVE_ROUND、ACTIVE_CEIL、ACTIVE_FLOOR额外支持FLOAT16，其余同BM1684。
   *
   */
  if (!(active_type == ACTIVE_ISINF || active_type == ACTIVE_ISNAN))
    SGDNN_CHECK(input.dtype == output.dtype);

  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));

  sg_api_active_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dim = input.dim;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  api.active_type = active_type;

#if defined BACKEND_1684X
  if (active_type == ACTIVE_ABSVAL) {
    SGDNN_CHECK(
        input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
        input.dtype == SGDNN_DTYPE_BF16);

  } else if (active_type == ACTIVE_RELU  ||
      active_type == ACTIVE_ROUND || active_type == ACTIVE_CEIL ||
      active_type == ACTIVE_FLOOR) {
    SGDNN_CHECK(
        input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
        input.dtype == SGDNN_DTYPE_INT8 || input.dtype == SGDNN_DTYPE_UINT8 ||
        input.dtype == SGDNN_DTYPE_INT16 || input.dtype == SGDNN_DTYPE_UINT16);
  } else if (active_type == ACTIVE_ERF || active_type == ACTIVE_ERFC ||
      active_type == ACTIVE_COSH || active_type == ACTIVE_SINH || active_type == ACTIVE_TANH) {
    SGDNN_CHECK(
        input.dtype == SGDNN_DTYPE_FP32);
  }
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_active", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_active_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLog(tpu_resource_t resource , SgdnnTensor_t input,
                     SgdnnTensor_t output, sg_log_type_t log_type,
                     bool non_blocking ) {

  if (log_type == LOG_E) {
    SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16);
  } else {
    SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32);
  }
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  sg_api_log_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  api.log_type = log_type;
#if defined BACKEND_1684X
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_log", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_log_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnSqueeze(tpu_resource_t resource , SgdnnTensor_t input,
                     SgdnnTensor_t output,
                     bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_FP16 ||
              input.dtype == SGDNN_DTYPE_BF16 ||
              input.dtype == SGDNN_DTYPE_INT32 ||
              input.dtype == SGDNN_DTYPE_INT64 );

  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  sg_api_squeeze_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
#if defined BACKEND_1684X
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_squeeze", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_squeeze_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNativeGroupNorm(tpu_resource_t resource , SgdnnTensor_t input,
                     SgdnnTensor_t gamma, SgdnnTensor_t beta,
                     int group, int affine, float eps, SgdnnTensor_t output,
                     SgdnnTensor_t mean, SgdnnTensor_t rstd,
                     bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_FP16 ||
              input.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  sg_api_native_group_norm_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.weight_global_addr = gamma.addr;
  api.bias_global_addr = beta.addr;
  api.mean_global_addr = mean.addr;
  api.rstd_global_addr = rstd.addr;
  api.group_num = group;
  api.eps = eps;
  api.affine = affine;
  api.axis = 1;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
#if defined BACKEND_1684X
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_native_group_norm", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_native_group_norm_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNativeGroupNormBackward(tpu_resource_t resource , SgdnnTensor_t grad_output,
                     SgdnnTensor_t input, SgdnnTensor_t weight,
                     SgdnnTensor_t mean, SgdnnTensor_t rstd,
                     int group, SgdnnTensor_t out0,
                     SgdnnTensor_t out1, SgdnnTensor_t out2,
                     bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_FP16 ||
              input.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  sg_api_groupnorm2d_backward_t api;
  api.input_global_addr = input.addr;
  api.weight_global_addr = weight.addr;
  api.grad_output_global_addr = grad_output.addr;
  api.group_nums = group;
  api.saved_mean_global_addr = mean.addr;
  api.saved_invstd_global_addr = rstd.addr;
  api.group_nums = group;
  api.grad_input_global_addr = out0.addr;
  api.grad_weight_global_addr = out1.addr;
  api.grad_bias_global_addr = out2.addr;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
#if defined BACKEND_1684X
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_groupnorm2d_backward", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_native_group_norm_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLogicalOr ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output ,
                              bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  // SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
  //               input.dtype == SGDNN_DTYPE_FP16 ||
  //               input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_logical_or_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_logical_or", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_logical_or_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_logical_or_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFlip ( tpu_resource_t resource ,
                         SgdnnTensor_t input,
                         int axis,
                         SgdnnTensor_t output ,
                         bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
#if defined BACKEND_1684X
  sg_api_flip_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.axis = axis;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_flip", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_flip_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.axis = axis;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_flip_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLogicalAnd ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output,
                              bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_logical_and_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_logical_and", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_logical_and_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_logical_and_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLogicalNot ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t output ,
                              bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_logical_not_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_logical_not", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_logical_not_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_logical_not_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}


tpu_status_t sgdnnBinary ( tpu_resource_t resource ,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            float scalar,
                            SgdnnTensor_t output,
                            int binary_type,
                            bool non_blocking )
{
  if ((input.format_casted || other.format_casted) && (output.addr == input.addr || output.addr == other.addr) ) { // this branch special for weight-grad update
    //printf("SGDNN Binary todo imple formated binary. input.format = %d, other.format = %d", input.format_casted, other.format_casted);
    SGDNN_CHECK( (input.format_casted == SGDNN_CONV_W_TRAIN_FORMAT && other.format_casted == SGDNN_CONV_DW_TRAIN_FORMAT) ||
                 (input.format_casted == SGDNN_CONV_DW_TRAIN_FORMAT && other.format_casted == SGDNN_CONV_W_TRAIN_FORMAT) );
    #if defined BACKEND_1684X
    sg_api_weight_update_t api;
    api.input_global_addr = input.addr;
    api.other_global_addr = other.addr;
    api.output_global_addr = output.addr;
    api.in_dim = input.dim;
    for ( int i = 0; i < input.dim; ++i )
    {
      api.in_shape[i] = input.shape[i];
    }
    api.other_dim = other.dim;
    for ( int i = 0; i < other.dim; ++i )
    {
      api.other_shape[i] = other.shape[i];
    }
    api.input_format = (int)input.format_casted;
    api.other_format = (int)other.format_casted;
    api.dtype = sgdnnTPUKernelDType ( input.dtype );
    api.value = scalar;
    api.binary_type = (binary_type == BINARY_ADD && scalar!=0 ) ? BINARY_ADDCMUL : binary_type;
      SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_weight_update", &api, sizeof ( api ) ) );
    #elif defined BACKEND_SG2260
      SGDNN_CHECK ( false );
    #else
      SGDNN_CHECK ( false );
    #endif
    return SG_SUCCESS;
  }

  SGDNN_CHECK ( input.dtype == other.dtype );
  // SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype != SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  sg_api_binary_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  api.binary_type = binary_type;
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << 2 <<std::endl;
  std::cout << "output_num\n" << 1 <<std::endl;
  std::cout << "input_addr0\n" << api.input_global_addr << std::endl;
  std::cout << "input_size0\n" << sgdnnTensorBytes(&input) << std::endl;
  std::cout << "input_addr1\n" << api.other_global_addr << std::endl;
  std::cout << "input_size1\n" << sgdnnTensorBytes(&other) << std::endl;
  std::cout << "output_addr0\n" << api.output_global_addr << std::endl;
  std::cout << "output_size0\n" << sgdnnTensorBytes(&output) << std::endl;
#endif
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_binary", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_binary_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnBinaryBcast (  tpu_resource_t resource ,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t other,
                                  float scalar,
                                  SgdnnTensor_t output,
                                  int binary_type ,
                                  bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == other.dtype );
  // SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype != SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( input.dim == other.dim );
  SGDNN_CHECK ( input.dim == output.dim );
  for ( int i = 0; i < input.dim; ++i )
  {
    SGDNN_CHECK ( input.shape[i] == 1 || input.shape[i] == output.shape[i] );
  }
  for ( int i = 0; i < other.dim; ++i )
  {
    SGDNN_CHECK ( other.shape[i] == 1 || other.shape[i] == output.shape[i] );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_binary_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i )
  {
    api.other_shape[i] = other.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  api.binary_type = binary_type;
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_binary_bcast", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_binary_bcast_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnBinaryC ( tpu_resource_t resource ,
                             SgdnnTensor_t input,
                             float scalar,
                             SgdnnTensor_t output,
                             int binary_type,
                             bool inversed ,
                             bool non_blocking )
{
  // SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype != SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  sg_api_binary_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  api.binary_type = binary_type;
  api.inversed = inversed;
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_binary_c", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_binary_c_multi_core", &api, sizeof ( api ), non_blocking ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}


tpu_status_t sgdnnCrossEntropyLoss ( tpu_resource_t resource ,
                                    SgdnnTensor_t input,
                                    SgdnnTensor_t target,
                                    int reduction,
                                    float label_smoothing,
                                    SgdnnTensor_t output ,
                                    bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( target.dtype == SGDNN_DTYPE_INT32 ||
                target.dtype == SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( input.dim == 2 );
  SGDNN_CHECK ( target.dim == 1 );
  SGDNN_CHECK ( output.dim == 0 );
  SGDNN_CHECK ( target.shape[0] == input.shape[0] );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &target ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  SGDNN_CHECK ( reduction == 0 || reduction == 1 );
#if defined BACKEND_1684X
  sg_api_cross_entropy_loss_t api;
  api.input_global_addr = input.addr;
  api.target_global_addr = target.addr;
  api.output_global_addr = output.addr;
  api.batch = input.shape[0];
  api.class_ = input.shape[1];
  api.reduction = reduction;
  api.label_smoothing = label_smoothing;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.is_target_int64 = target.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_cross_entropy_loss", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_cross_entropy_loss_t api;
  api.input_global_addr = input.addr;
  api.target_global_addr = target.addr;
  api.output_global_addr = output.addr;
  api.batch = input.shape[0];
  api.class_ = input.shape[1];
  api.reduction = reduction;
  api.label_smoothing = label_smoothing;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.is_target_int64 = target.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_cross_entropy_loss_multicore", &api, sizeof ( api ) , non_blocking) );

#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnCrossEntropyLossBackward (
tpu_resource_t resource ,
SgdnnTensor_t input,
SgdnnTensor_t target,
SgdnnTensor_t grad_output,
int reduction,
float label_smoothing,
SgdnnTensor_t grad_input ,
bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  SGDNN_CHECK ( input.dtype == grad_input.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( target.dtype == SGDNN_DTYPE_INT32 ||
                target.dtype == SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( input.dim == 2 );
  SGDNN_CHECK ( target.dim == 1 );
  SGDNN_CHECK ( grad_output.dim == 0 );
  SGDNN_CHECK ( grad_input.dim == 2 );
  SGDNN_CHECK ( target.shape[0] == input.shape[0] );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &target ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  SGDNN_CHECK ( reduction == 0 || reduction == 1 );
#if defined BACKEND_1684X
  sg_api_cross_entropy_loss_backward_t api;
  api.input_global_addr = input.addr;
  api.target_global_addr = target.addr;
  api.grad_output_global_addr = grad_output.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.batch = input.shape[0];
  api.class_ = input.shape[1];
  api.reduction = reduction;
  api.label_smoothing = label_smoothing;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.is_target_int64 = target.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_cross_entropy_loss_backward", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_cross_entropy_loss_backward_t api;
  api.input_global_addr = input.addr;
  api.target_global_addr = target.addr;
  api.grad_output_global_addr = grad_output.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.batch = input.shape[0];
  api.class_ = input.shape[1];
  api.reduction = reduction;
  api.label_smoothing = label_smoothing;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.is_target_int64 = target.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_cross_entropy_loss_backward_multicore", &api, sizeof ( api ) , non_blocking) );

#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLLamaA16Matmul ( tpu_resource_t handle,
                             SgdnnTensor_t left,
                             SgdnnTensor_t right,
                             SgdnnTensor_t bias,
                             SgdnnTensor_t scale,
                             SgdnnTensor_t zp,
                             int group_size,
                             int weight_bits,
                             SgdnnTensor_t output,
                             bool non_blocking )
{
#if defined BACKEND_SG2260
  if ( left.dtype == SGDNN_DTYPE_FP16 || left.dtype == SGDNN_DTYPE_BF16)
  {
    sg_api_a16_matmul_t api;
    api.input_global_addr = left.addr;
    api.weight_global_addr = right.addr;
    api.bias_global_addr = bias.addr;
    api.scale_global_addr = scale.addr;
    api.zp_global_addr = zp.addr;
    api.output_global_addr = output.addr;
    api.R_trans = true;
    api.sign = true;
    api.has_bias = int(bias.addr != 0);
    api.has_zp = true;
    api.final_row_num = left.shape[left.dim-2];
    api.inner_num = left.shape[left.dim-1];
    api.final_col_num = right.shape[left.dim-2];
    api.q_group_size = abs(group_size);
    api.weight_bits = weight_bits;
    api.io_dtype = sgdnnTPUKernelDType ( left.dtype );
    api.weight_dtype = sgdnnTPUKernelDType (right.dtype);
    if (bias.addr != 0) {
      api.bias_dtype = sgdnnTPUKernelDType (bias.dtype);
    }
    SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( handle, "tpu_kernel_api_llama_a16_matmul", &api, sizeof ( api ), non_blocking ) );
    return SG_SUCCESS;
  } else {
    SGDNN_CHECK ( false );
  }
#else
  SGDNN_CHECK ( false );
#endif
}

tpu_status_t sgdnnEmbeddingBackward ( tpu_resource_t resource ,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t indices,
                                     SgdnnTensor_t grad_input ,
                                     bool non_blocking )
{
  SGDNN_CHECK ( grad_output.dtype == grad_input.dtype );
  SGDNN_CHECK ( grad_output.dtype == SGDNN_DTYPE_FP32 ||
                grad_output.dtype == SGDNN_DTYPE_FP16 ||
                grad_output.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( indices.dtype == SGDNN_DTYPE_INT32 || indices.dtype == SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( grad_input.dim == 2 );
  SGDNN_CHECK ( grad_output.shape[grad_output.dim - 1] == grad_input.shape[1] );
#if defined BACKEND_1684X
  sg_api_embedding_backward_t api;
  const int window_size = 64;
  SGDNN_CHECK ( window_size % 64 == 0 );
  int indices_num = 1;
  for ( int i = 0; i < indices.dim; ++i )
  {
    indices_num *= indices.shape[i];
  }
  tpu_device_mem_t sorted_index, sorted_index_index, from_index, to_index;
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &sorted_index, indices_num * sizeof ( int ) ) );
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &sorted_index_index, indices_num * sizeof ( int ) ) );
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &from_index, window_size * sizeof ( int ) ) );
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &to_index, window_size * sizeof ( int ) ) );
  api.grad_output_global_addr = grad_output.addr;
  api.index_global_addr = indices.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.sorted_index_global_addr = sgdnnGetDeviceAddr ( sorted_index );
  api.sorted_index_index_global_addr = sgdnnGetDeviceAddr ( sorted_index_index );
  api.from_index_global_addr = sgdnnGetDeviceAddr ( from_index );
  api.to_index_global_addr = sgdnnGetDeviceAddr ( to_index );
  api.grad_output_dim = grad_output.dim;
  for ( int i = 0; i < grad_output.dim; ++i )
  {
    api.grad_output_shape[i] = grad_output.shape[i];
  }
  api.index_dim = indices.dim;
  for ( int i = 0; i < indices.dim; ++i )
  {
    api.index_shape[i] = indices.shape[i];
  }
  api.grad_input_dim = grad_input.dim;
  for ( int i = 0; i < grad_input.dim; ++i )
  {
    api.grad_input_shape[i] = grad_input.shape[i];
  }
  api.window_size = window_size;
  api.grad_output_dtype = sgdnnTPUKernelDType ( grad_output.dtype );
  api.is_index_int64 = indices.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_embedding_backward", &api, sizeof ( api ) ) );
  sgdnnFreeDevice ( resource , sorted_index );
  sgdnnFreeDevice ( resource , sorted_index_index );
  sgdnnFreeDevice ( resource , from_index );
  sgdnnFreeDevice ( resource , to_index );
#elif defined BACKEND_SG2260
  sg_api_embedding_backward_t api;
  const int window_size = 64;
  SGDNN_CHECK ( window_size % 64 == 0 );
  int indices_num = 1;
  for ( int i = 0; i < indices.dim; ++i )
  {
    indices_num *= indices.shape[i];
  }
  tpu_device_mem_t sorted_index, sorted_index_index, from_index, to_index;
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &sorted_index, indices_num * sizeof ( int ) ) );
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &sorted_index_index, indices_num * sizeof ( int ) ) );
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &from_index, window_size * sizeof ( int ) ) );
  SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &to_index, window_size * sizeof ( int ) ) );
  api.grad_output_global_addr = grad_output.addr;
  api.index_global_addr = indices.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.sorted_index_global_addr = sgdnnGetDeviceAddr ( sorted_index );
  api.sorted_index_index_global_addr = sgdnnGetDeviceAddr ( sorted_index_index );
  api.from_index_global_addr = sgdnnGetDeviceAddr ( from_index );
  api.to_index_global_addr = sgdnnGetDeviceAddr ( to_index );
  api.grad_output_dim = grad_output.dim;
  for ( int i = 0; i < grad_output.dim; ++i )
  {
    api.grad_output_shape[i] = grad_output.shape[i];
  }
  api.index_dim = indices.dim;
  for ( int i = 0; i < indices.dim; ++i )
  {
    api.index_shape[i] = indices.shape[i];
  }
  api.grad_input_dim = grad_input.dim;
  for ( int i = 0; i < grad_input.dim; ++i )
  {
    api.grad_input_shape[i] = grad_input.shape[i];
  }
  api.window_size = window_size;
  api.grad_output_dtype = sgdnnTPUKernelDType ( grad_output.dtype );
  api.is_index_int64 = indices.dtype == SGDNN_DTYPE_INT64;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_embedding_backward_multicore", &api, sizeof ( api ), non_blocking ) );
  sgdnnFreeDevice ( resource , sorted_index );
  sgdnnFreeDevice ( resource , sorted_index_index );
  sgdnnFreeDevice ( resource , from_index );
  sgdnnFreeDevice ( resource , to_index );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNorm2 ( tpu_resource_t resource ,
                         SgdnnTensor_t input,
                         int keepdim,
                         SgdnnTensor_t output ,
                         bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( keepdim )
  {
    SGDNN_CHECK ( input.dim == output.dim );
    for ( int i = 0; i < output.dim; ++i )
    {
      SGDNN_CHECK ( output.shape[i] == 1 );
    }
  }
  else
  {
    SGDNN_CHECK ( (size_t)output.dim == 0 );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_norm2_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_norm2", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_norm2_multi_core_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  // make a buffer to store middle results FP32
  unsigned int size = 8 * sizeof(float);

  tpu_device_mem_t dev_mem;
  tpu_status_t err = sgdnnMallocDeviceByte(resource , &dev_mem, size);
  if(SG_SUCCESS != err){
    printf("malloc device error \r\n");
    return err;
  }
  api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_norm2_multi_core", &api, sizeof ( api ) , non_blocking) );

#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnGELU ( tpu_resource_t resource ,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output ,
                        bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  sg_api_gelu_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << 1 <<std::endl;
  std::cout << "output_num\n" << 1 <<std::endl;
  std::cout << "input_addr0\n" << api.input_global_addr << std::endl;
  std::cout << "input_size0\n" << sgdnnTensorBytes(&input) << std::endl;
  std::cout << "output_addr0\n" << api.output_global_addr << std::endl;
  std::cout << "output_size0\n" << sgdnnTensorBytes(&output) << std::endl;
#endif
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_gelu", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_gelu_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnReLUBackward ( tpu_resource_t resource ,
                                SgdnnTensor_t grad_output,
                                SgdnnTensor_t input,
                                SgdnnTensor_t grad_input ,
                                bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  SGDNN_CHECK ( input.dtype == grad_input.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_output ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
#if defined BACKEND_1684X
  sg_api_relu_backward_t api;
  api.input_global_addr = input.addr;
  api.grad_output_global_addr = grad_output.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_relu_backward", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnGELUBackward ( tpu_resource_t resource ,
                                SgdnnTensor_t grad_output,
                                SgdnnTensor_t input,
                                SgdnnTensor_t grad_input ,
                                bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  SGDNN_CHECK ( input.dtype == grad_input.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_output ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  sg_api_gelu_backward_t api;
  api.input_global_addr = input.addr;
  api.grad_output_global_addr = grad_output.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
#ifdef USING_PERF_MODE
  std::cout << "input_num\n" << 2 <<std::endl;
  std::cout << "output_num\n" << 1 <<std::endl;
  std::cout << "input_addr0\n" << api.grad_output_global_addr << std::endl;
  std::cout << "input_size0\n" << sgdnnTensorBytes(&grad_output) << std::endl;
  std::cout << "input_addr1\n" << api.input_global_addr << std::endl;
  std::cout << "input_size1\n" << sgdnnTensorBytes(&input) << std::endl;
  std::cout << "output_addr0\n" << api.grad_input_global_addr << std::endl;
  std::cout << "output_size0\n" << sgdnnTensorBytes(&grad_input) << std::endl;
#endif
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_gelu_backward", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_gelu_backward_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLeakyReLU ( tpu_resource_t resource ,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output,
                       float negative_slope ,
                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
#if defined BACKEND_1684X
  sg_api_leakyrelu_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.negative_slope = negative_slope;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_leakyrelu", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_leakyrelu_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.negative_slope = negative_slope;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_leakyrelu_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnSign ( tpu_resource_t resource ,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output ,
                        bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_sign_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_sign", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_active_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.active_type = ACTIVE_SIGN;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "active_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAddCMulBcast ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output ,
                           bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == tensor1.dtype );
  SGDNN_CHECK ( input.dtype == tensor2.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &tensor1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &tensor2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_bcast_addcmul_t api;
  api.input_global_addr = input.addr;
  api.tensor1_global_addr = tensor1.addr;
  api.tensor2_global_addr = tensor2.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.tensor1_dim = tensor1.dim;
  api.tensor2_dim = tensor2.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < tensor1.dim; ++i )
  {
    api.tensor1_shape[i] = tensor1.shape[i];
  }
  for ( int i = 0; i < tensor2.dim; ++i )
  {
    api.tensor2_shape[i] = tensor2.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_addcmul_bcast", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  //SGDNN_CHECK ( false );
  sg_api_bcast_addcmul_t api;
  api.input_global_addr = input.addr;
  api.tensor1_global_addr = tensor1.addr;
  api.tensor2_global_addr = tensor2.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.tensor1_dim = tensor1.dim;
  api.tensor2_dim = tensor2.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < tensor1.dim; ++i )
  {
    api.tensor1_shape[i] = tensor1.shape[i];
  }
  for ( int i = 0; i < tensor2.dim; ++i )
  {
    api.tensor2_shape[i] = tensor2.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_addcmul_bcast", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAddCMul ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output ,
                           bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == tensor1.dtype );
  SGDNN_CHECK ( input.dtype == tensor2.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &tensor1 ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &tensor2 ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &tensor1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &tensor2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_addcmul_t api;
  api.input_global_addr = input.addr;
  api.tensor1_global_addr = tensor1.addr;
  api.tensor2_global_addr = tensor2.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_addcmul", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  //SGDNN_CHECK ( false );
  sg_api_addcmul_t api;
  api.input_global_addr = input.addr;
  api.tensor1_global_addr = tensor1.addr;
  api.tensor2_global_addr = tensor2.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "addcmul_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAddCDiv ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output ,
                           bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == tensor1.dtype );
  SGDNN_CHECK ( input.dtype == tensor2.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &tensor1 ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &tensor2 ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &tensor1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &tensor2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_addcdiv_t api;
  api.input_global_addr = input.addr;
  api.tensor1_global_addr = tensor1.addr;
  api.tensor2_global_addr = tensor2.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_addcdiv", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_addcdiv_t api;
  api.input_global_addr = input.addr;
  api.tensor1_global_addr = tensor1.addr;
  api.tensor2_global_addr = tensor2.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "addcdiv_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMaskedFill ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t mask,
                              float value,
                              SgdnnTensor_t out ,
                              bool non_blocking )
{
  SGDNN_CHECK ( input.dim == mask.dim );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 ||
                input.dtype == SGDNN_DTYPE_INT32 );
  for ( int i = 0; i < input.dim; i++ )
  {
    SGDNN_CHECK ( input.shape[i] == mask.shape[i] || mask.shape[i] == 1);
  }
#if defined BACKEND_1684X
  sg_api_masked_fill_t api;
  api.input_global_addr = input.addr;
  api.mask_global_addr = mask.addr;
  api.out_global_addr = out.addr;
  api.input_dims = input.dim;
  api.mask_dims = mask.dim;
  for ( int i = 0; i < input.dim; i++ )
  {
    api.input_shape[i] = input.shape[i];
    api.mask_shape[i] = mask.shape[i];
  }
  api.value = value;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_masked_fill", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_masked_fill_t api;
  api.input_global_addr = input.addr;
  api.mask_global_addr = mask.addr;
  api.out_global_addr = out.addr;
  api.input_dims = input.dim;
  api.mask_dims = mask.dim;
  for ( int i = 0; i < input.dim; i++ )
  {
    api.input_shape[i] = input.shape[i];
    api.mask_shape[i] = mask.shape[i];
  }
  api.value = value;
  api.dtype = sgdnnTPUKernelDType(input.dtype);//sgdnnTPUKernelDType ( SGDNN_DTYPE_FP32 );;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_masked_fill_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnDropout ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           unsigned long long seed,
                           float threshold,
                           SgdnnTensor_t output,
                           SgdnnTensor_t mask ,
                           bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  if ( mask.addr != 0 )
  {
    SGDNN_CHECK ( mask.dtype == SGDNN_DTYPE_UINT8 );
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  if ( mask.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsSameShape ( &input, &mask ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  if ( mask.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &mask ) );
  }
#if defined BACKEND_1684X
  SGDNN_CHECK ( false ); // TODO
  sg_api_dropout_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.mask_global_addr = mask.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.seed = seed;
  api.threshold = threshold;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_dropout", &api, sizeof ( api ) ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMlp ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w1,
                          SgdnnTensor_t w2,
                          SgdnnTensor_t b1,
                          SgdnnTensor_t b2,
                          SgdnnTensor_t out1,
                          SgdnnTensor_t p,
                          SgdnnTensor_t output ,
                          bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == w1.dtype );
  SGDNN_CHECK ( input.dtype == w2.dtype );
  SGDNN_CHECK ( input.dtype == out1.dtype );
  SGDNN_CHECK ( input.dtype == p.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( b1.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == b1.dtype );
  }
  if ( b2.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == b2.dtype );
  }

  SGDNN_CHECK ( input.shape[2] == w1.shape[0] );
  SGDNN_CHECK ( out1.shape[2] == w1.shape[1] );
  SGDNN_CHECK ( sgdnnIsSameShape( &out1, &p ) );
  SGDNN_CHECK ( p.shape[2] == w2.shape[0] );
  SGDNN_CHECK ( output.shape[2] == w2.shape[1] );
  SGDNN_CHECK ( input.shape[0] == output.shape[0] );
  SGDNN_CHECK ( input.shape[1] == output.shape[1] );

  if ( b1.addr != 0 )
  {
    SGDNN_CHECK ( b1.dim == 1 );
    SGDNN_CHECK ( b1.shape[0] == w1.shape[1] );
    SGDNN_CHECK ( b1.shape[0] == out1.shape[2] );
  }

  if ( b2.addr != 0 )
  {
    SGDNN_CHECK ( b2.dim == 1 );
    SGDNN_CHECK ( b2.shape[0] == w2.shape[1] );
    SGDNN_CHECK ( b2.shape[0] == output.shape[2] );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &p ) );

  if ( b1.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &b1 ) );
  }
  if ( b2.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &b2 ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  sg_api_mlp_multi_core_t api;
  api.input_addr    = input.addr;
  api.weight0_addr  = w1.addr;
  api.weight1_addr  = w2.addr;
  api.bias0_addr    = b1.addr;
  api.bias1_addr    = b2.addr;
  api.output_addr   = output.addr;
  api.in_dims       = input.dim;
  api.w0_dims       = w1.dim;
  api.w1_dims       = w2.dim;
  api.has_bias      = (int)(b1.addr != 0)  + 2 * (int)(b2.addr != 0);
  api.use_fast      = 0;
  api.in_dtype      = sgdnnTPUKernelDType ( input.dtype );
  api.out_dtype     = sgdnnTPUKernelDType ( output.dtype );

  for ( int i = 0; i < input.dim; ++i )
  {
    api.in_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < w1.dim; ++i )
  {
    api.w0_shape[i] = w1.shape[i];
  }
  for ( int i = 0; i < w2.dim; ++i )
  {
    api.w1_shape[i] = w2.shape[i];
  }
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_mlp_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}


tpu_status_t sgdnnMlpBackward ( tpu_resource_t resource ,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t w1,
                                  SgdnnTensor_t w2,
                                  SgdnnTensor_t out1,
                                  SgdnnTensor_t p,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_w1,
                                  SgdnnTensor_t grad_w2,
                                  SgdnnTensor_t grad_b1,
                                  SgdnnTensor_t grad_b2,
                                  bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == w1.dtype );
  SGDNN_CHECK ( input.dtype == w2.dtype );
  SGDNN_CHECK ( input.dtype == out1.dtype );
  SGDNN_CHECK ( input.dtype == p.dtype );
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  if ( grad_input.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_input.dtype );
  }
  if ( grad_w1.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_w1.dtype );
  }
  if ( grad_b1.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_b1.dtype );
  }
  if ( grad_w2.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_w2.dtype );
  }
  if ( grad_b2.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_b2.dtype );
  }

  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );

  SGDNN_CHECK ( input.shape[0] == grad_output.shape[0] );
  SGDNN_CHECK ( input.shape[1] == grad_output.shape[1] );
  SGDNN_CHECK ( sgdnnIsSameShape( &out1, &p ) );
  SGDNN_CHECK ( input.shape[2] == w1.shape[0] );
  SGDNN_CHECK ( out1.shape[2] == w1.shape[1] );
  SGDNN_CHECK ( p.shape[2] == w2.shape[0] );

  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &w1, &grad_w1 ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &w2, &grad_w2 ) );

  if ( grad_b1.addr != 0 )
  {
    SGDNN_CHECK ( grad_b1.shape[0] == out1.shape[2] );
  }
  if ( grad_b2.addr != 0 )
  {
    SGDNN_CHECK ( grad_b2.shape[0] == grad_w2.shape[1] );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &p ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_w1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_w2 ) );

  if ( grad_b1.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_b1 ) );
  }
  if ( grad_b2.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_b2 ) );
  }

#if defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLLamaMlp ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight0,
                          SgdnnTensor_t weight1,
                          SgdnnTensor_t weight2,
                          SgdnnTensor_t output,
                          int group_num,
                          int block_num,
                          bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == weight0.dtype );
  SGDNN_CHECK ( input.dtype == weight1.dtype );
  SGDNN_CHECK ( input.dtype == weight2.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );

  SGDNN_CHECK ( sgdnnIsSameShape( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsSameShape( &weight0, &weight1 ) );
  SGDNN_CHECK ( input.dim == 2 );
  SGDNN_CHECK ( weight0.dim == 2 );
  SGDNN_CHECK ( weight2.dim == 2 );
  SGDNN_CHECK ( weight0.shape[0] == input.shape[1] );
  SGDNN_CHECK ( weight2.shape[0] == weight0.shape[1] );

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight0 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  sg_api_llama_mlp_multi_core_t api;
  api.input_addr    = input.addr;
  api.weight0_addr  = weight0.addr;
  api.weight1_addr  = weight1.addr;
  api.weight2_addr  = weight2.addr;
  api.output_addr   = output.addr;
  api.batch         = input.shape[0];
  api.input_w       = input.shape[1];
  api.middle_w      = weight0.shape[1];
  api.dtype         = sgdnnTPUKernelDType ( input.dtype );
  api.quantized      = false;

  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_llama_mlp_multi_core", &api, sizeof ( api ) , non_blocking, group_num, block_num ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLLamaA16Mlp ( tpu_resource_t  resource ,
                                     SgdnnTensor_t input,
                                     SgdnnTensor_t weight0,
                                     SgdnnTensor_t zp0,
                                     SgdnnTensor_t scale0,
                                     SgdnnTensor_t weight1,
                                     SgdnnTensor_t zp1,
                                     SgdnnTensor_t scale1,
                                     SgdnnTensor_t weight2,
                                     SgdnnTensor_t zp2,
                                     SgdnnTensor_t scale2,
                                     int group_size,
                                     int weight_bits,
                                     SgdnnTensor_t output,
                                     int group_num,
                                     int block_num,
                                     bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == scale0.dtype );
  SGDNN_CHECK ( input.dtype == scale1.dtype );
  SGDNN_CHECK ( input.dtype == scale2.dtype );
  SGDNN_CHECK ( weight0.dtype == zp0.dtype );
  SGDNN_CHECK ( weight1.dtype == zp1.dtype );
  SGDNN_CHECK ( weight2.dtype == zp2.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP16 || input.dtype == SGDNN_DTYPE_BF16);

  SGDNN_CHECK ( sgdnnIsSameShape( &input, &output ) );
  SGDNN_CHECK ( input.dim == 2 );
  SGDNN_CHECK ( weight0.dim == 2 );
  SGDNN_CHECK ( zp0.dim == 2 );
  SGDNN_CHECK ( scale0.dim == 2 );
  SGDNN_CHECK ( weight2.dim == 2 );
  SGDNN_CHECK ( zp2.dim == 2 );
  SGDNN_CHECK ( scale2.dim == 2 );

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight0 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &zp0 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &scale0 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &zp1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &scale1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &zp2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &scale2 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  sg_api_llama_mlp_multi_core_t api;
  api.input_addr    = input.addr;
  api.weight0_addr  = weight0.addr;
  api.zp0_addr  = zp0.addr;
  api.scale0_addr  = scale0.addr;
  api.weight1_addr  = weight1.addr;
  api.zp1_addr  = zp1.addr;
  api.scale1_addr  = scale1.addr;
  api.weight2_addr  = weight2.addr;
  api.zp2_addr  = zp2.addr;
  api.scale2_addr  = scale2.addr;
  api.output_addr   = output.addr;
  api.batch         = input.shape[0];
  api.input_w       = input.shape[1];
  api.middle_w      = weight0.shape[0];
  api.dtype         = sgdnnTPUKernelDType ( input.dtype );
  api.quantized     = true;
  api.group_size  = group_size;
  api.weight_bits   = weight_bits;
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource, "tpu_kernel_llama_mlp_multi_core", &api, sizeof ( api ) , non_blocking, group_num, block_num ) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLlamaAttention ( tpu_resource_t resource,
                                  SgdnnTensor_t OUT,
                                  SgdnnTensor_t Q,
                                  SgdnnTensor_t K,
                                  SgdnnTensor_t V,
                                  SgdnnTensor_t Kcache,
                                  SgdnnTensor_t Vcache,
                                  SgdnnTensor_t cos,
                                  SgdnnTensor_t sin,
                                  SgdnnTensor_t input_lengths,
                                  SgdnnTensor_t save_slots,
                                  SgdnnTensor_t fetch_slots,
                                  SgdnnTensor_t mask,
                                  int slots_size,
                                  int mask_size,
                                  int block_size,
                                  float C,
                                  int attention_mode,
                                  bool non_blocking)
{
  SGDNN_CHECK ( Q.dtype == OUT.dtype );
  SGDNN_CHECK ( Q.dtype == K.dtype );
  SGDNN_CHECK ( Q.dtype == V.dtype );
  SGDNN_CHECK ( Q.dtype == Kcache.dtype );
  SGDNN_CHECK ( Q.dtype == Vcache.dtype );
  SGDNN_CHECK ( Q.dtype == SGDNN_DTYPE_FP32 ||
                Q.dtype == SGDNN_DTYPE_FP16 ||
                Q.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape( &Q, &OUT ) );
  SGDNN_CHECK ( sgdnnIsSameShape( &K, &V ) );
  SGDNN_CHECK ( Q.dim == 3 );
  SGDNN_CHECK ( K.dim == 3 );
  SGDNN_CHECK ( V.dim == 3 );
  SGDNN_CHECK ( Kcache.dim == 4 );
  SGDNN_CHECK ( Vcache.dim == 4 );
  if (cos.addr != 0){
    SGDNN_CHECK ( Q.dtype == cos.dtype );
    SGDNN_CHECK ( Q.dtype == sin.dtype );
    SGDNN_CHECK ( cos.dim == 3 );
    SGDNN_CHECK ( sin.dim == 3 );
    SGDNN_CHECK ( sgdnnIsSameShape( &cos, &sin ) );
  }
  if (mask.addr != 0){
    SGDNN_CHECK ( Q.dtype == mask.dtype );
    SGDNN_CHECK ( mask.dim == 2 );
  }

#ifndef USE_QKV_PACKED
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &Q ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &K ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &V ) );
#endif

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &Kcache ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &Vcache ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &OUT ) );

#if defined BACKEND_1684X
  // sg_api_llama2_qkv_t api;
  // api.OUT_global_addr = OUT.addr;
  // api.Q_global_addr = Q.addr;
  // api.K_global_addr = K.addr;
  // api.V_global_addr = V.addr;
  // api.Kcache_global_addr = Kcache.addr;
  // api.Vcache_global_addr = Vcache.addr;
  // api.input_lengths_global_addr = input_lengths.addr;
  // api.save_slots_global_addr = save_slots.addr;
  // api.fetch_slots_global_addr = fetch_slots.addr;
  // api.mask_global_addr = mask.addr;
  // api.slots_size = slots_size;
  // api.mask_size = mask_size;
  // api.block_size = block_size;
  // api.C = C;

  // api.attention_mode = attention_mode;
  // if (attention_mode == 2){
  //   // prefill
  //   api.attention_mode = 1;
  // } else if (attention_mode == 3){
  //   // decode
  //   api.attention_mode = 0;
  // }

  // api.dtype = sgdnnTPUKernelDType(Q.dtype);
  // api.batch = input_lengths.shape[0];
  // api.head_size = Q.shape[2];
  // api.q_heads = Q.shape[1];
  // api.kv_heads = K.shape[1];
  // SAFE_CALL ( sgdnnTPUKernelLaunch ( resource, "tpu_kernel_llama_attention", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
//   sg_api_llama2_qkv_multi_core_t api = {0};
//   api.OUT_global_addr = OUT.addr;
//   api.Q_global_addr = Q.addr;
//   api.K_global_addr = K.addr;
//   api.V_global_addr = V.addr;
//   api.Kcache_global_addr = Kcache.addr;
//   api.Vcache_global_addr = Vcache.addr;
//   api.cos_global_addr = cos.addr;
//   api.sin_global_addr = sin.addr;
//   api.input_lengths_global_addr = input_lengths.addr;
//   api.save_slots_global_addr = save_slots.addr;
//   api.fetch_slots_global_addr = fetch_slots.addr;
//   api.mask_global_addr = mask.addr;
//   api.slots_size = slots_size;
//   api.mask_size = mask_size;
//   api.block_size = block_size;
//   api.C = C;
//   api.attention_mode = attention_mode;
//   api.dtype = sgdnnTPUKernelDType(Q.dtype);
//   api.batch = input_lengths.shape[0];
//   api.hidden_size = Q.shape[1] * Q.shape[2]; // heads * head_size
//   api.q_heads = Q.shape[1];
//   api.kv_heads = K.shape[1];
// #ifdef USE_QKV_PACKED
//   api.qkv_packed = !sgdnnIsTensorContiguous(&Q);
// #endif

//   tpu_device_mem_t Qbuffer_dev_mem, Kbuffer_dev_mem, Vbuffer_dev_mem;
//   if (attention_mode == 2)
//   {
//     // prefill
//     SAFE_CALL(sgdnnMallocDeviceByte(resource, &Qbuffer_dev_mem, sgdnnTensorBytes(&Q)));
//     SAFE_CALL(sgdnnMallocDeviceByte(resource, &Kbuffer_dev_mem, sgdnnTensorBytes(&K)));
//     SAFE_CALL(sgdnnMallocDeviceByte(resource, &Vbuffer_dev_mem, sgdnnTensorBytes(&V)));
//   }
//   api.Qbuffer_global_addr = sgdnnGetDeviceAddr(Qbuffer_dev_mem);
//   api.Kbuffer_global_addr = sgdnnGetDeviceAddr(Kbuffer_dev_mem);
//   api.Vbuffer_global_addr = sgdnnGetDeviceAddr(Vbuffer_dev_mem);
//   SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource, "tpu_kernel_llama_attention_multi_core", &api, sizeof ( api ) , non_blocking) );
//   if (attention_mode == 2){
//     sgdnnFreeDevice ( resource , Kbuffer_dev_mem );
//     sgdnnFreeDevice ( resource , Vbuffer_dev_mem );
//     sgdnnFreeDevice ( resource , Qbuffer_dev_mem );
//   }
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLlamaAttentionForward ( tpu_resource_t resource,
                                  SgdnnTensor_t OUT,
                                  SgdnnTensor_t Q,
                                  SgdnnTensor_t K,
                                  SgdnnTensor_t V,
                                  SgdnnTensor_t cos,
                                  SgdnnTensor_t sin,
                                  SgdnnTensor_t mask,
                                  SgdnnTensor_t softmax_lse,
                                  int mask_size,
                                  float C,
                                  float dropout_rate,
                                  int batch,
                                  bool non_blocking)
{
  SGDNN_CHECK ( Q.dtype == OUT.dtype );
  SGDNN_CHECK ( Q.dtype == K.dtype );
  SGDNN_CHECK ( Q.dtype == V.dtype );
  SGDNN_CHECK ( Q.dtype == SGDNN_DTYPE_FP32 ||
                Q.dtype == SGDNN_DTYPE_FP16 ||
                Q.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape( &Q, &OUT ) );
  SGDNN_CHECK ( sgdnnIsSameShape( &K, &V ) );
  SGDNN_CHECK ( Q.dim == 3 );
  SGDNN_CHECK ( K.dim == 3 );
  SGDNN_CHECK ( V.dim == 3 );
  SGDNN_CHECK ( batch > 0 );
  if (cos.addr != 0){
    SGDNN_CHECK ( Q.dtype == cos.dtype );
    SGDNN_CHECK ( Q.dtype == sin.dtype );
    SGDNN_CHECK ( cos.dim == 3 );
    SGDNN_CHECK ( sin.dim == 3 );
    SGDNN_CHECK ( sgdnnIsSameShape( &cos, &sin ) );
  }
  if (mask.addr != 0){
    SGDNN_CHECK ( Q.dtype == mask.dtype );
    SGDNN_CHECK ( mask.dim == 2 );
  }

#ifndef USE_QKV_PACKED
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &Q ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &K ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &V ) );
#endif

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &OUT ) );

#if defined BACKEND_1684X
#elif defined BACKEND_SG2260
  sg_api_llama_attention_forward_multi_core_t api = {0};
  api.Y_global_addr = OUT.addr;
  api.Q_global_addr = Q.addr;
  api.K_global_addr = K.addr;
  api.V_global_addr = V.addr;
  api.cos_global_addr = cos.addr;
  api.sin_global_addr = sin.addr;
  api.mask_global_addr = mask.addr;
  api.Softmax_lse_global_addr = softmax_lse.addr;
  api.mask_max = mask_size;
  api.C = C;
  api.dropout_rate = dropout_rate;
  api.dtype = sgdnnTPUKernelDType(Q.dtype);
  api.batch = batch;
  api.hidden_size = Q.shape[1] * Q.shape[2]; // heads * head_size
  api.num_attention_heads = Q.shape[1];
  api.num_k_v_heads = K.shape[1];
  api.seq_len = Q.shape[0] / batch;
  api.return_softmax = softmax_lse.addr != 0;
#ifdef USE_QKV_PACKED
  api.qkv_packed = !sgdnnIsTensorContiguous(&Q);
#endif

  tpu_device_mem_t Qbuffer_dev_mem, Kbuffer_dev_mem, Vbuffer_dev_mem;

  // prefill
  SAFE_CALL(sgdnnMallocDeviceByte(resource, &Qbuffer_dev_mem, sgdnnTensorBytes(&Q)));
  SAFE_CALL(sgdnnMallocDeviceByte(resource, &Kbuffer_dev_mem, sgdnnTensorBytes(&K)));
  SAFE_CALL(sgdnnMallocDeviceByte(resource, &Vbuffer_dev_mem, sgdnnTensorBytes(&V)));

  api.Qbuffer_global_addr = sgdnnGetDeviceAddr(Qbuffer_dev_mem);
  api.Kbuffer_global_addr = sgdnnGetDeviceAddr(Kbuffer_dev_mem);
  api.Vbuffer_global_addr = sgdnnGetDeviceAddr(Vbuffer_dev_mem);
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource, "tpu_kernel_llama_attention_forward_multi_core", &api, sizeof ( api ) , non_blocking) );
  sgdnnFreeDevice ( resource , Kbuffer_dev_mem );
  sgdnnFreeDevice ( resource , Vbuffer_dev_mem );
  sgdnnFreeDevice ( resource , Qbuffer_dev_mem );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnRMSNorm ( tpu_resource_t  resource ,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight,
                          SgdnnTensor_t bias,
                          SgdnnTensor_t output,
                          int           axis,
                          float         eps,
                          float         partial,
                          int           with_scale,
                          int           with_bias,
                          bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );

  SGDNN_CHECK ( sgdnnIsSameShape( &input, &output ) );
  SGDNN_CHECK ( axis >= 0 && axis < input.dim );

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  sg_api_rmsnorm_multi_core_t api;
  api.input_addr    = input.addr;
  api.weight_addr   = weight.addr;
  api.bias_addr     = bias.addr;
  api.output_addr   = output.addr;
  memcpy(api.shape, input.shape, sizeof(int) * input.dim);
  api.dims          = input.dim;
  api.axis          = axis;
  api.eps           = eps;
  api.with_weight   = with_scale;
  api.with_bias     = with_bias;
  api.partial       = partial;
  api.dtype         = sgdnnTPUKernelDType ( input.dtype );

  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_rmsnorm_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPMU(tpu_resource_t resource, int enable, bool non_blocking)
{
#if defined BACKEND_SG2260
  sg_api_pmu_t api;
  api.enable = enable;
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource, "tpu_kernel_pmu", &api, sizeof(api), non_blocking));
  if(!enable) {
    getPMU(resource);
  }
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAttn ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w_attn,
                          SgdnnTensor_t w_proj,
                          SgdnnTensor_t b_attn,
                          SgdnnTensor_t b_proj,
                          SgdnnTensor_t q,
                          SgdnnTensor_t k,
                          SgdnnTensor_t v,
                          SgdnnTensor_t softmax_out,
                          SgdnnTensor_t soft_v,
                          SgdnnTensor_t out ,
                          bool non_blocking )
{
  SGDNN_CHECK ( input.dim == 3 );
  SGDNN_CHECK ( q.dim == 3 );
  SGDNN_CHECK ( k.dim == 3 );
  SGDNN_CHECK ( v.dim == 3 );
  SGDNN_CHECK ( softmax_out.dim == 4 );
  SGDNN_CHECK ( soft_v.dim == 4 );
  SGDNN_CHECK ( out.dim == 3 );

  SGDNN_CHECK ( input.dtype == w_attn.dtype );
  SGDNN_CHECK ( input.dtype == w_proj.dtype );
  SGDNN_CHECK ( input.dtype == q.dtype );
  SGDNN_CHECK ( input.dtype == k.dtype );
  SGDNN_CHECK ( input.dtype == v.dtype );
  SGDNN_CHECK ( input.dtype == softmax_out.dtype );
  SGDNN_CHECK ( input.dtype == soft_v.dtype );
  SGDNN_CHECK ( input.dtype == out.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( b_attn.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == b_attn.dtype );
  }
  if ( b_proj.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == b_proj.dtype );
  }

  SGDNN_CHECK ( input.shape[2] == w_attn.shape[0] );
  SGDNN_CHECK ( q.shape[2] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( k.shape[2] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( v.shape[2] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( softmax_out.shape[2] == input.shape[1] && softmax_out.shape[3] == input.shape[1] );
  SGDNN_CHECK ( soft_v.shape[1] == softmax_out.shape[1] && soft_v.shape[2] == softmax_out.shape[2] );
  SGDNN_CHECK ( soft_v.shape[3] == q.shape[2] / soft_v.shape[1] );
  SGDNN_CHECK ( w_proj.shape[0] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( w_proj.shape[1] == input.shape[2] );
  SGDNN_CHECK ( out.shape[2] == w_proj.shape[1] );
  SGDNN_CHECK ( sgdnnIsSameShape( &input, &out ) );

  if ( b_attn.addr != 0 )
  {
    SGDNN_CHECK ( b_attn.dim == 1 );
    SGDNN_CHECK ( b_attn.shape[0] == w_attn.shape[1] );
  }

  if ( b_proj.addr != 0 )
  {
    SGDNN_CHECK ( b_proj.dim == 1 );
    SGDNN_CHECK ( b_proj.shape[0] == w_proj.shape[1] );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w_attn ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w_proj ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &q ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &k ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &v ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &softmax_out ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &soft_v ) );

  if ( b_attn.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &b_attn ) );
  }
  if ( b_proj.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &b_proj ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}


tpu_status_t sgdnnAttnBackward ( tpu_resource_t resource ,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t w_attn,
                                  SgdnnTensor_t w_proj,
                                  SgdnnTensor_t q,
                                  SgdnnTensor_t k,
                                  SgdnnTensor_t v,
                                  SgdnnTensor_t softmax_out,
                                  SgdnnTensor_t soft_v,
                                  SgdnnTensor_t bias,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_w_attn,
                                  SgdnnTensor_t grad_w_proj,
                                  SgdnnTensor_t grad_b_attn,
                                  SgdnnTensor_t grad_b_proj,
                                  bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == w_attn.dtype );
  SGDNN_CHECK ( input.dtype == w_proj.dtype );
  SGDNN_CHECK ( input.dtype == q.dtype );
  SGDNN_CHECK ( input.dtype == k.dtype );
  SGDNN_CHECK ( input.dtype == v.dtype );
  SGDNN_CHECK ( input.dtype == softmax_out.dtype );
  SGDNN_CHECK ( input.dtype == soft_v.dtype );
  SGDNN_CHECK ( input.dtype == grad_output.dtype );
  SGDNN_CHECK ( input.dtype == grad_input.dtype );

  SGDNN_CHECK ( input.dtype == grad_w_attn.dtype );
  SGDNN_CHECK ( input.dtype == grad_w_proj.dtype );
  if ( grad_b_attn.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_b_attn.dtype );
  }
  if ( grad_b_proj.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == grad_b_proj.dtype );
  }

  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );

  SGDNN_CHECK ( input.shape[2] == w_attn.shape[0] );
  SGDNN_CHECK ( q.shape[2] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( k.shape[2] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( v.shape[2] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( softmax_out.shape[2] == input.shape[1] && softmax_out.shape[3] == input.shape[1] );
  SGDNN_CHECK ( soft_v.shape[1] == softmax_out.shape[1] && soft_v.shape[2] == softmax_out.shape[2] );
  SGDNN_CHECK ( soft_v.shape[3] == q.shape[2] / soft_v.shape[1] );
  SGDNN_CHECK ( w_proj.shape[0] == w_attn.shape[1]/3 );
  SGDNN_CHECK ( w_proj.shape[1] == input.shape[2] );
  SGDNN_CHECK ( bias.shape[2] == input.shape[1] );

  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &grad_input ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &w_attn, &grad_w_attn ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &w_proj, &grad_w_proj ) );
  if ( grad_b_attn.addr != 0 )
  {
    SGDNN_CHECK ( grad_b_attn.shape[0] == grad_w_attn.shape[1] );
  }
  if ( grad_b_proj.addr != 0 )
  {
    SGDNN_CHECK ( grad_b_proj.shape[0] == grad_w_proj.shape[1] );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w_attn ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w_proj ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &q ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &k ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &v ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &softmax_out ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &soft_v ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &bias ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_output ) );

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_w_attn ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_w_proj ) );
  if ( grad_b_attn.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_b_attn ) );
  }
  if ( grad_b_proj.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &grad_b_proj ) );
  }

#if defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnArange ( tpu_resource_t resource ,
                                int start,
                                int end,
                                int step,
                                SgdnnTensor_t out,
                                bool non_blocking )
{
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
  SGDNN_CHECK( out.dim == 1);  // arange just need 1 dimension.
  SGDNN_CHECK( out.dtype == SGDNN_DTYPE_INT32 || out.dtype == SGDNN_DTYPE_INT64 || out.dtype == SGDNN_DTYPE_FP32 );
  sg_api_arange_t api;
  api.start = start;
  api.end = end;
  api.step = step;
  api.output_global_addr = out.addr;
  api.dim = out.dim;
  api.shape[0] = out.shape[0];
  if ( out.dtype == SGDNN_DTYPE_INT64 ) {
    api.dtype =  sgdnnTPUKernelDType ( SGDNN_DTYPE_INT32 );
    api.isint64 = 1;
  } else {
    api.dtype =  sgdnnTPUKernelDType ( out.dtype );
    api.isint64 = 0;
  }
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_arange", &api, sizeof ( api ) , non_blocking) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource, "tpu_kernel_api_arange_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnElementBitwise ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              int mode,
                              SgdnnTensor_t output ,
                              bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_INT32 ||
                input.dtype == SGDNN_DTYPE_UINT32 ||
                input.dtype == SGDNN_DTYPE_INT16 ||
                input.dtype == SGDNN_DTYPE_UINT16 ||
                input.dtype == SGDNN_DTYPE_INT8 ||
                input.dtype == SGDNN_DTYPE_UINT8 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_element_bitwise_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.shape[i] = input.shape[i];
  }
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );

#if defined BACKEND_1684X
   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_element_bitwise", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_element_bitwise_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnElementBitwiseBcast ( tpu_resource_t resource ,
                                   SgdnnTensor_t input,
                                   SgdnnTensor_t other,
                                   int mode,
                                   SgdnnTensor_t output ,
                                   bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_INT32 ||
                input.dtype == SGDNN_DTYPE_UINT32 ||
                input.dtype == SGDNN_DTYPE_INT16 ||
                input.dtype == SGDNN_DTYPE_UINT16 ||
                input.dtype == SGDNN_DTYPE_INT8 ||
                input.dtype == SGDNN_DTYPE_UINT8 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_element_bitwise_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i ) {
    api.other_shape[i] = other.shape[i];
  }
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );

#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_element_bitwise_bcast", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_element_bitwise_bcast_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnElementBitwiseC ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              int scalar,
                              int mode,
                              SgdnnTensor_t output,
                              bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_INT32 ||
                input.dtype == SGDNN_DTYPE_UINT32 ||
                input.dtype == SGDNN_DTYPE_INT16 ||
                input.dtype == SGDNN_DTYPE_UINT16 ||
                input.dtype == SGDNN_DTYPE_INT8 ||
                input.dtype == SGDNN_DTYPE_UINT8 );
  SGDNN_CHECK ( input.dim == output.dim );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ));
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_element_bitwise_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  api.mode = mode;

#if defined BACKEND_1684X
  SAFE_CALL( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_element_bitwise_c", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_element_bitwise_c_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnNeg ( tpu_resource_t resource ,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output ,
                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  // SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32);

  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_neg_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_neg", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_neg_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
   SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_neg_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnBitwiseNot(tpu_resource_t resource , SgdnnTensor_t input,
                            SgdnnTensor_t output,
                            bool non_blocking ) {
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_INT32);

  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_bitwise_not_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_bitwise_not", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_bitwise_not_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(
      resource , "tpu_kernel_api_bitwise_not_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnComparision ( tpu_resource_t resource ,
                               SgdnnTensor_t input,
                               SgdnnTensor_t other,
                               int mode,
                               SgdnnTensor_t output ,
                               bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  SgdnnTensor_t input_ = input, other_ = other;
  tpu_device_mem_t input_mem, other_mem;
  if (sgdnnDtypeIsInt64(input.dtype))
  {
    SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &input_mem, sgdnnTensorBytes ( &input ) / 2 ) );
    SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &other_mem, sgdnnTensorBytes ( &other ) / 2 ) );
    input_.dtype = SGDNN_DTYPE_INT32;
    other_.dtype = SGDNN_DTYPE_INT32;
    input_.addr = sgdnnGetDeviceAddr ( input_mem );
    other_.addr = sgdnnGetDeviceAddr ( other_mem );
    sgdnnConvertInt64toInt32(resource , input, input_);
    sgdnnConvertInt64toInt32(resource , other, other_);
  }
  sg_api_comparision_t api;
  api.input_global_addr = input_.addr;
  api.other_global_addr = other_.addr;
  api.output_global_addr = output.addr;
  api.dim = input_.dim;
  for(int i = 0; i < input_.dim; ++i) {
    api.shape[i] = input_.shape[i];
  }
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input_.dtype);
#if defined BACKEND_1684X
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_comparision", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_comparision_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK ( false );
#endif
  if (sgdnnDtypeIsInt64(input.dtype)){
    sgdnnFreeDevice ( resource , input_mem );
    sgdnnFreeDevice ( resource , other_mem );
  }
  return SG_SUCCESS;
}

tpu_status_t sgdnnComparisionBcast ( tpu_resource_t resource ,
                                    SgdnnTensor_t input,
                                    SgdnnTensor_t other,
                                    int mode,
                                    SgdnnTensor_t output ,
                                    bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_comparision_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i ) {
    api.other_shape[i] = other.shape[i];
  }
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );

#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_comparision_bcast", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_comparision_bcast_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnComparisionC ( tpu_resource_t resource ,
                                SgdnnTensor_t input,
                                float scalar,
                                int mode,
                                int scalar_pos,
                                SgdnnTensor_t output ,
                                bool non_blocking ) {
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_comparision_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.const_value = scalar;
  api.mode = mode;
  api.scalar_pos = scalar_pos;
  api.dtype = sgdnnTPUKernelDType(input.dtype);

#if defined BACKEND_1684X
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_comparision_c", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_comparision_c_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnShiftLeftC ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          char const_value,
                          SgdnnTensor_t output,
                          bool non_blocking ) {
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  sg_api_shift_left_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.const_value = const_value;
  api.src_dtype = sgdnnTPUKernelDType(input.dtype);
  api.dst_dtype = sgdnnTPUKernelDType(output.dtype);
  api.other_dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_shift_left_c", &api, sizeof(api), non_blocking));
  return SG_SUCCESS;
}

tpu_status_t sgdnnShiftLeft ( tpu_resource_t resource ,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  sg_api_shift_left_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.src_dtype = sgdnnTPUKernelDType(input.dtype);
  api.dst_dtype = sgdnnTPUKernelDType(output.dtype);
  api.other_dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_shift_left", &api, sizeof(api), non_blocking));
  return SG_SUCCESS;
}

tpu_status_t sgdnnShiftLeftBcast ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output ,
                              bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dim == other.dim );
  SGDNN_CHECK ( input.dim == output.dim );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  for ( int i = 0; i < input.dim; ++i ) {
    SGDNN_CHECK ( input.shape[i] == 1 || input.shape[i] == output.shape[i] );
  }
  for ( int i = 0; i < other.dim; ++i ) {
    SGDNN_CHECK ( other.shape[i] == 1 || other.shape[i] == output.shape[i] );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_shift_left_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i ) {
    api.other_shape[i] = other.shape[i];
  }
  api.src_dtype = sgdnnTPUKernelDType(input.dtype);
  api.dst_dtype = sgdnnTPUKernelDType(output.dtype);
  api.other_dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_shift_left_bcast", &api, sizeof ( api ) , non_blocking) );

  return SG_SUCCESS;
}

tpu_status_t sgdnnShiftRightArithmetic ( tpu_resource_t resource ,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_shift_right_arithmetic_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.src_dtype = sgdnnTPUKernelDType(input.dtype);
  api.dst_dtype = sgdnnTPUKernelDType(output.dtype);
  api.other_dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_shift_right_arithmetic", &api, sizeof(api), non_blocking));

  return SG_SUCCESS;
}

tpu_status_t sgdnnShiftRightArithmeticC ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          int const_value,
                          SgdnnTensor_t output,
                          bool non_blocking ) {
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_shift_right_arithmetic_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.const_value = const_value;
  api.src_dtype = sgdnnTPUKernelDType(input.dtype);
  api.dst_dtype = sgdnnTPUKernelDType(output.dtype);
  api.other_dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_shift_right_arithmetic_c", &api, sizeof(api), non_blocking));

  return SG_SUCCESS;
}

tpu_status_t sgdnnMinimumC(tpu_resource_t resource , SgdnnTensor_t input, float scalar,
                          SgdnnTensor_t output,
                          bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_minimumc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_minimumc", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_minimumc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_minimumc_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMinimum(tpu_resource_t resource , SgdnnTensor_t input,
                         SgdnnTensor_t other, SgdnnTensor_t output,
                         bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &other));
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_minimum_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_minimum", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_minimum_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_minimum_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMinimumBcast(tpu_resource_t resource , SgdnnTensor_t input,
                              SgdnnTensor_t other, SgdnnTensor_t output,
                              bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_minimum_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_minimum_bcast", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_minimum_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(
      resource , "tpu_kernel_api_minimum_bcast_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMaximumC(tpu_resource_t resource , SgdnnTensor_t input, float scalar,
                          SgdnnTensor_t output,
                          bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_maximumc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_maximumc", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_maximumc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_maximumc_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMaximum(tpu_resource_t resource , SgdnnTensor_t input,
                         SgdnnTensor_t other, SgdnnTensor_t output,
                         bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &other));
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_maximum_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_maximum", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_maximum_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_maximum_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnMaximumBcast(tpu_resource_t resource , SgdnnTensor_t input,
                              SgdnnTensor_t other, SgdnnTensor_t output,
                              bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_maximum_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_maximum_bcast", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_maximum_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(
      resource , "tpu_kernel_api_maximum_bcast_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAtan2C(tpu_resource_t resource , float scalar, SgdnnTensor_t other,
                        SgdnnTensor_t output,
                        bool non_blocking ) {
  SGDNN_CHECK(other.dtype == SGDNN_DTYPE_FP32);
  SGDNN_CHECK(output.dtype == SGDNN_DTYPE_FP32 ||
              output.dtype == SGDNN_DTYPE_FP16 ||
              output.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(sgdnnIsSameShape(&other, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_atan2c_t api;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = other.dim;
  api.scalar = scalar;
  for (int i = 0; i < other.dim; ++i) {
    api.shape[i] = other.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_atan2c", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_atan2c_t api;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = other.dim;
  api.scalar = scalar;
  for (int i = 0; i < other.dim; ++i) {
    api.shape[i] = other.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_atan2c_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAtan2_C(tpu_resource_t resource , SgdnnTensor_t input, float scalar,
                         SgdnnTensor_t output,
                         bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32);
  SGDNN_CHECK(output.dtype == SGDNN_DTYPE_FP32 ||
              output.dtype == SGDNN_DTYPE_FP16 ||
              output.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_atan2_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_atan2_c", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_atan2_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_atan2_c_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAtan2(tpu_resource_t resource , SgdnnTensor_t input,
                       SgdnnTensor_t other, SgdnnTensor_t output,
                       bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32);
  SGDNN_CHECK(output.dtype == SGDNN_DTYPE_FP32 ||
              output.dtype == SGDNN_DTYPE_FP16 ||
              output.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &other));
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_atan2_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_atan2", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_atan2_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_atan2_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAtan2Bcast(tpu_resource_t resource , SgdnnTensor_t input,
                            SgdnnTensor_t other, SgdnnTensor_t output,
                            bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32);
  SGDNN_CHECK(output.dtype == SGDNN_DTYPE_FP32 ||
              output.dtype == SGDNN_DTYPE_FP16 ||
              output.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(input.dtype == other.dtype);

  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_atan2_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_atan2_bcast", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_atan2_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(
      resource , "tpu_kernel_api_atan2_bcast_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnBaddbmm ( tpu_resource_t resource ,
                          SgdnnTensor_t input1,
                          SgdnnTensor_t batch1,
                          SgdnnTensor_t batch2,
                          SgdnnTensor_t out,
                          double alpha,
                          double beta,
                          bool non_blocking )
{
  SGDNN_CHECK ( input1.dtype == SGDNN_DTYPE_FP32||
                input1.dtype == SGDNN_DTYPE_FP16||
                input1.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input1.dtype == batch1.dtype );
  SGDNN_CHECK ( input1.dtype == batch2.dtype );
  SGDNN_CHECK ( input1.dtype == out.dtype );
  // shape check
  SGDNN_CHECK ( batch1.dim == 3 );
  SGDNN_CHECK ( batch2.dim == 3 );
  int batch_out_shape[8] = {0};
  batch_out_shape[0] = batch1.shape[0];
  batch_out_shape[1] = batch1.shape[1];
  batch_out_shape[2] = batch2.shape[2];
  // check boardcast
  int min_dim = input1.dim;
  for ( int i = 0; i < min_dim; ++i )
  {
    SGDNN_CHECK ( batch_out_shape[2-i] == input1.shape[min_dim-1-i] );
  }
  // input1 must can be broadcast to batch_out_shape
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
#if defined BACKEND_1684X
  // need malloc buffer to store input1
  tpu_device_mem_t input1_buffer, left_contiguous_mem;
  SgdnnTensor_t left_contiguous;
  if ( beta != 0){
    SAFE_CALL( sgdnnMallocDeviceByte ( resource , &input1_buffer, sgdnnTensorBytes ( &input1 ) ) );
  }
  if ( sgdnnIsTensorTransposed (&batch1) ){
    left_contiguous.dim = 3;
    left_contiguous.shape[0] = batch1.shape[0];
    left_contiguous.shape[1] = batch1.shape[1];
    left_contiguous.shape[2] = batch1.shape[2];
    left_contiguous.stride[0] = batch1.stride[0];
    left_contiguous.stride[1] = batch1.stride[2];
    left_contiguous.stride[2] = batch1.stride[1];
    left_contiguous.dtype = batch1.dtype;
    SAFE_CALL ( sgdnnMallocDeviceByte ( resource , &left_contiguous_mem, sgdnnTensorBytes ( &left_contiguous ) ) );
    left_contiguous.addr = sgdnnGetDeviceAddr ( left_contiguous_mem );
    SAFE_CALL ( sgdnnStridedCopy ( resource , batch1, left_contiguous) );
  }
  sg_api_baddbmm_t api;
  api.input_global_addr  = input1.addr;
  api.buffer_global_addr = beta == 0 ? 0x0 : input1_buffer.u.device.device_addr;
  api.batch1_global_addr = sgdnnIsTensorTransposed (&batch1) ? left_contiguous.addr : batch1.addr ;
  api.batch2_global_addr = batch2.addr;
  api.output_global_addr = out.addr;
  api.input_dim = input1.dim;
  api.batch1_dim = batch1.dim;
  api.batch2_dim = batch2.dim;
  api.output_dim = out.dim;
  for ( int i = 0; i < input1.dim; ++i )
  {
    api.input_shape[i] = input1.shape[i];
  }
  for ( int i = 0; i < batch1.dim; ++i )
  {
    api.batch1_shape[i] = batch1.shape[i];
  }
  for ( int i = 0; i < batch2.dim; ++i )
  {
    api.batch2_shape[i] = batch2.shape[i];
  }
  for ( int i = 0; i < out.dim; ++i )
  {
    api.output_shape[i] = out.shape[i];
  }
  api.alpha = (float)alpha;
  api.beta = (float)beta;
  api.dtype = sgdnnTPUKernelDType ( input1.dtype );
  api.is_left_transpose = 0;
  api.is_right_transpose = sgdnnIsTensorTransposed( &batch2 );

  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_baddbmm", &api, sizeof ( api ) ) );
  if (beta != 0){
    sgdnnFreeDevice(resource , input1_buffer);
  }
  if ( sgdnnIsTensorTransposed (&batch1) )
  {
    sgdnnFreeDevice ( resource , left_contiguous_mem );
  }

#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
return SG_SUCCESS;
}

tpu_status_t sgdnnMseloss( tpu_resource_t resource ,
                                    SgdnnTensor_t self,
                                    SgdnnTensor_t target,
                                    SgdnnTensor_t out,
                                    int reduction ,
                                    bool non_blocking ) {
  SGDNN_CHECK ( sgdnnIsSameShape ( &self, &target ) );
  if ( reduction == 0 )
  {
    SGDNN_CHECK ( sgdnnIsSameShape ( &self, &out ) );
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &self ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &target ) );

  #if defined BACKEND_1684X
    sg_api_mse_loss_t api;
    api.input1_global_addr = self.addr;
    api.input2_global_addr = target.addr;
    api.output_global_addr = out.addr;
    api.reduction = reduction;
    api.dim = self.dim;
  for(int i = 0; i < self.dim; ++i) {
    api.shape[i] = self.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(self.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_mse_loss", &api, sizeof(api)));
  #elif defined BACKEND_SG2260
    // SGDNN_CHECK ( false );
      sg_api_mse_loss_t api;
      api.input1_global_addr = self.addr;
      api.input2_global_addr = target.addr;
      api.output_global_addr = out.addr;
      api.reduction = reduction;
      api.dim = self.dim;
    for(int i = 0; i < self.dim; ++i) {
      api.shape[i] = self.shape[i];
    }
    api.dtype = sgdnnTPUKernelDType(self.dtype);
    SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_mse_loss_multi_core", &api, sizeof(api), non_blocking));
  #else
  SGDNN_CHECK ( false );
  #endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnLnMm ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w,
                          SgdnnTensor_t b,
                          SgdnnTensor_t gamma,
                          SgdnnTensor_t beta,
                          float eps,
                          SgdnnTensor_t mean,
                          SgdnnTensor_t rstd,
                          SgdnnTensor_t output,
                          bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == w.dtype );
  SGDNN_CHECK ( input.dtype == gamma.dtype );
  SGDNN_CHECK ( input.dtype == beta.dtype );
  SGDNN_CHECK ( input.dtype == mean.dtype );
  SGDNN_CHECK ( input.dtype == rstd.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  if ( b.addr != 0 )
  {
    SGDNN_CHECK ( input.dtype == b.dtype );
  }

  const auto input_ndim = input.dim;
  SGDNN_CHECK ( input_ndim == 2 || input_ndim == 3);
  if ( input_ndim == 2 )
  {
    SGDNN_CHECK ( input.shape[1] == w.shape[0] );
    SGDNN_CHECK ( input.shape[1] == gamma.shape[0] );
    SGDNN_CHECK ( sgdnnIsSameShape( &gamma, &beta ) );
    SGDNN_CHECK ( sgdnnIsSameShape( &mean, &rstd ) );
    SGDNN_CHECK ( input.shape[0] == mean.shape[0] );
    SGDNN_CHECK ( output.shape[1] == w.shape[1] );
    SGDNN_CHECK ( input.shape[0] == output.shape[0] );
  } else {
    SGDNN_CHECK ( input.shape[2] == w.shape[0] );
    SGDNN_CHECK ( input.shape[2] == gamma.shape[0] );
    SGDNN_CHECK ( sgdnnIsSameShape( &gamma, &beta ) );
    SGDNN_CHECK ( sgdnnIsSameShape( &mean, &rstd ) );
    SGDNN_CHECK ( input.shape[0] == mean.shape[0] );
    SGDNN_CHECK ( input.shape[1] == mean.shape[1] );
    SGDNN_CHECK ( output.shape[2] == w.shape[1] );
    SGDNN_CHECK ( input.shape[0] == output.shape[0] );
    SGDNN_CHECK ( input.shape[1] == output.shape[1] );
  }

  if ( b.addr != 0 )
  {
    SGDNN_CHECK ( b.dim == 1 );
    SGDNN_CHECK ( b.shape[0] == w.shape[1] );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &gamma ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &beta ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &mean ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &rstd ) );

  if ( b.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &b ) );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  sg_api_ln_mm_multi_core_t api;
  api.input_addr    = input.addr;
  api.weight_addr   = w.addr;
  api.bias_addr     = b.addr;
  api.gamma_addr    = gamma.addr;
  api.beta_addr     = beta.addr;
  api.mean_addr     = mean.addr;
  api.rstd_addr     = rstd.addr;
  api.output_addr   = output.addr;
  api.in_dims       = input.dim;
  api.w_dims        = w.dim;
  api.eps           = eps;
  api.has_bias      = (int)(b.addr != 0);
  api.in_dtype      = sgdnnTPUKernelDType ( input.dtype );

  for ( int i = 0; i < input.dim; ++i )
  {
    api.in_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < w.dim; ++i )
  {
    api.w_shape[i] = w.shape[i];
  }

  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_ln_mm_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFmaxC(tpu_resource_t resource , SgdnnTensor_t input, float scalar,
                       SgdnnTensor_t output,
                       bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_fmaxc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_fmaxc", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_fmaxc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_fmaxc_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFmax(tpu_resource_t resource , SgdnnTensor_t input,
                      SgdnnTensor_t other, SgdnnTensor_t output,
                      bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &other));
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_fmax_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_fmax", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_fmax_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_fmax_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFmaxBcast(tpu_resource_t resource , SgdnnTensor_t input,
                           SgdnnTensor_t other, SgdnnTensor_t output,
                           bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_fmax_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_fmax_bcast", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_fmax_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_fmax_bcast_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFminC(tpu_resource_t resource , SgdnnTensor_t input, float scalar,
                       SgdnnTensor_t output,
                       bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_fminc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_fminc", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_fminc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_fminc_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFmin(tpu_resource_t resource , SgdnnTensor_t input,
                      SgdnnTensor_t other, SgdnnTensor_t output,
                      bool non_blocking) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &other));
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_fmin_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_fmin", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_fmin_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_fmin_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnFminBcast(tpu_resource_t resource , SgdnnTensor_t input,
                           SgdnnTensor_t other, SgdnnTensor_t output,
                           bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_fmin_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_fmin_bcast", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_fmin_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_fmin_bcast_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnAddLnMm ( tpu_resource_t resource ,
                          SgdnnTensor_t input0,
                          SgdnnTensor_t input1,
                          SgdnnTensor_t w,
                          SgdnnTensor_t b,
                          SgdnnTensor_t gamma,
                          SgdnnTensor_t beta,
                          float eps,
                          SgdnnTensor_t out_add,
                          SgdnnTensor_t mean,
                          SgdnnTensor_t rstd,
                          SgdnnTensor_t output ,
                          bool non_blocking )
{
  SGDNN_CHECK ( input0.dtype == input1.dtype );
  SGDNN_CHECK ( input0.dtype == w.dtype );
  SGDNN_CHECK ( input0.dtype == gamma.dtype );
  SGDNN_CHECK ( input0.dtype == beta.dtype );
  SGDNN_CHECK ( input0.dtype == out_add.dtype );
  SGDNN_CHECK ( input0.dtype == mean.dtype );
  SGDNN_CHECK ( input0.dtype == rstd.dtype );
  SGDNN_CHECK ( input0.dtype == output.dtype );
  SGDNN_CHECK ( input0.dtype == SGDNN_DTYPE_FP32 ||
                input0.dtype == SGDNN_DTYPE_FP16 ||
                input0.dtype == SGDNN_DTYPE_BF16 );
  if ( b.addr != 0 )
  {
    SGDNN_CHECK ( input0.dtype == b.dtype );
  }

  const auto input_ndim = input0.dim;
  SGDNN_CHECK ( input_ndim == 2 || input_ndim == 3);
  SGDNN_CHECK ( sgdnnIsSameShape( &input0, &input1 ) );
  SGDNN_CHECK ( sgdnnIsSameShape( &input0, &out_add ) );
  if ( input_ndim == 2 )
  {
    SGDNN_CHECK ( input0.shape[1] == w.shape[0] );
    SGDNN_CHECK ( input0.shape[1] == gamma.shape[0] );
    SGDNN_CHECK ( sgdnnIsSameShape( &gamma, &beta ) );
    SGDNN_CHECK ( sgdnnIsSameShape( &mean, &rstd ) );
    SGDNN_CHECK ( input0.shape[0] == mean.shape[0] );
    SGDNN_CHECK ( output.shape[1] == w.shape[1] );
    SGDNN_CHECK ( input0.shape[0] == output.shape[0] );
  } else {
    SGDNN_CHECK ( input0.shape[2] == w.shape[0] );
    SGDNN_CHECK ( input0.shape[2] == gamma.shape[0] );
    SGDNN_CHECK ( sgdnnIsSameShape( &gamma, &beta ) );
    SGDNN_CHECK ( sgdnnIsSameShape( &mean, &rstd ) );
    SGDNN_CHECK ( input0.shape[0] == mean.shape[0] );
    SGDNN_CHECK ( input0.shape[1] == mean.shape[1] );
    SGDNN_CHECK ( output.shape[2] == w.shape[1] );
    SGDNN_CHECK ( input0.shape[0] == output.shape[0] );
    SGDNN_CHECK ( input0.shape[1] == output.shape[1] );
  }

  if ( b.addr != 0 )
  {
    SGDNN_CHECK ( b.dim == 1 );
    SGDNN_CHECK ( b.shape[0] == w.shape[1] );
  }

  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input0 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input1 ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &w ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &gamma ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &beta ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out_add ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &mean ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &rstd ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  if ( b.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &b ) );
  }

#if defined BACKEND_1684X
  SGDNN_CHECK ( false );
#elif defined BACKEND_SG2260
  sg_api_add_ln_mm_multi_core_t api;
  api.input0_addr   = input0.addr;
  api.input1_addr   = input1.addr;
  api.weight_addr   = w.addr;
  api.bias_addr     = b.addr;
  api.gamma_addr    = gamma.addr;
  api.beta_addr     = beta.addr;
  api.out_add_addr  = out_add.addr;
  api.mean_addr     = mean.addr;
  api.rstd_addr     = rstd.addr;
  api.output_addr   = output.addr;
  api.in_dims       = input0.dim;
  api.w_dims        = w.dim;
  api.in_dtype      = sgdnnTPUKernelDType ( input0.dtype );
  api.eps           = eps;
  api.has_bias      = (int)(b.addr != 0);
  api.use_fast      = 0;

  for ( int i = 0; i < input0.dim; ++i )
  {
    api.in_shape[i] = input0.shape[i];
  }
  for ( int i = 0; i < w.dim; ++i )
  {
    api.w_shape[i] = w.shape[i];
  }

  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_add_ln_mm_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}


tpu_status_t sgdnnSignbit ( tpu_resource_t resource ,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output ,
                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32||
                input.dtype == SGDNN_DTYPE_FP16||
                input.dtype == SGDNN_DTYPE_BF16 );

  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_signbit_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_signbit", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_signbit_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_signbit_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPow ( tpu_resource_t resource ,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output ,
                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 ||
                input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK ( other.dtype == SGDNN_DTYPE_FP32 ||
                other.dtype == SGDNN_DTYPE_FP16 ||
                other.dtype == SGDNN_DTYPE_BF16 ||
                other.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_pow_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.input_dtype = sgdnnTPUKernelDType ( input.dtype );
  api.other_dtype = sgdnnTPUKernelDType ( other.dtype );
  api.output_dtype = sgdnnTPUKernelDType ( output.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_pow", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_pow_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.input_dtype = sgdnnTPUKernelDType ( input.dtype );
  api.other_dtype = sgdnnTPUKernelDType ( other.dtype );
  api.output_dtype = sgdnnTPUKernelDType ( output.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_pow_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPowBcast ( tpu_resource_t resource ,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            SgdnnTensor_t output,
                            bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 ||
                input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK ( other.dtype == SGDNN_DTYPE_FP32 ||
                other.dtype == SGDNN_DTYPE_FP16 ||
                other.dtype == SGDNN_DTYPE_BF16 ||
                other.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK ( input.dim == other.dim );
  SGDNN_CHECK ( input.dim == output.dim );
  for ( int i = 0; i < input.dim; ++i )
  {
    SGDNN_CHECK ( input.shape[i] == 1 || input.shape[i] == output.shape[i] );
  }
  for ( int i = 0; i < other.dim; ++i )
  {
    SGDNN_CHECK ( other.shape[i] == 1 || other.shape[i] == output.shape[i] );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined BACKEND_1684X
  sg_api_pow_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i )
  {
    api.other_shape[i] = other.shape[i];
  }
  api.input_dtype = sgdnnTPUKernelDType ( input.dtype );
  api.other_dtype = sgdnnTPUKernelDType ( other.dtype );
  api.output_dtype = sgdnnTPUKernelDType ( output.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_pow_bcast", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_pow_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i )
  {
    api.other_shape[i] = other.shape[i];
  }
  api.input_dtype = sgdnnTPUKernelDType ( input.dtype );
  api.other_dtype = sgdnnTPUKernelDType ( other.dtype );
  api.output_dtype = sgdnnTPUKernelDType ( output.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_pow_bcast_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPowC ( tpu_resource_t resource ,
                      SgdnnTensor_t self,
                      float scalar,
                      SgdnnTensor_t out ,
                      bool non_blocking )
{
  SGDNN_CHECK ( self.dtype == SGDNN_DTYPE_FP32 ||
                self.dtype == SGDNN_DTYPE_FP16 ||
                self.dtype == SGDNN_DTYPE_BF16 ||
                self.dtype == SGDNN_DTYPE_INT32);
  if (self.dtype == SGDNN_DTYPE_INT32) {
    SGDNN_CHECK(out.dtype == SGDNN_DTYPE_INT32 ||
                out.dtype == SGDNN_DTYPE_FP32);
  } else {
    SGDNN_CHECK(self.dtype == out.dtype);
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &self, &out ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &self ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
  sg_api_pow_tensor_scalar_t api;
  api.self_global_addr = self.addr;
  api.out_global_addr = out.addr;
  api.dim = self.dim;
  for ( int i = 0; i < self.dim; ++i )
  {
    api.shape[i] = self.shape[i];
  }
  api.value = scalar;
  api.dtype = sgdnnTPUKernelDType ( self.dtype );
  api.out_is_int = (out.dtype == SGDNN_DTYPE_INT32);
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_pow_c", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_pow_c_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnCPow ( tpu_resource_t resource ,
                      SgdnnTensor_t self,
                      float scalar,
                      SgdnnTensor_t out ,
                      bool non_blocking )
{
  // SGDNN_CHECK ( self.dtype == out.dtype );
  SGDNN_CHECK(self.dtype == SGDNN_DTYPE_FP32 ||
              self.dtype == SGDNN_DTYPE_INT32 ||
              self.dtype == SGDNN_DTYPE_BF16 ||
              self.dtype == SGDNN_DTYPE_FP16);
  if (self.dtype == SGDNN_DTYPE_INT32) {
    SGDNN_CHECK(out.dtype == SGDNN_DTYPE_INT32 ||
                out.dtype == SGDNN_DTYPE_FP32);
  } else {
    SGDNN_CHECK(self.dtype == out.dtype);
  }
  SGDNN_CHECK ( sgdnnIsSameShape ( &self, &out ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &self ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
  sg_api_pow_tensor_scalar_t api;
  api.self_global_addr = self.addr;
  api.out_global_addr = out.addr;
  api.dim = self.dim;
  for ( int i = 0; i < self.dim; ++i )
  {
    api.shape[i] = self.shape[i];
  }
  api.value = scalar;
  api.dtype = sgdnnTPUKernelDType ( self.dtype );
  api.out_is_int = (out.dtype == SGDNN_DTYPE_INT32);
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_c_pow", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_c_pow_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnReal ( tpu_resource_t resource ,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output ,
                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  // SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
#if defined BACKEND_1684X
  sg_api_real_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  for (int j = 0; j < input.dim; ++j )
  {
    api.input_stride[j] = input.stride[j];
    api.output_stride[j] = output.stride[j];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_real", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_real_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_real_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}
tpu_status_t sgdnnConj ( tpu_resource_t resource ,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output ,
                       bool non_blocking )
{
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  // SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
#if defined BACKEND_1684X
  sg_api_real_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  for (int j = 0; j < input.dim; ++j )
  {
    api.input_stride[j] = input.stride[j];
    api.output_stride[j] = output.stride[j];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conj", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_real_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_conj_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPermute ( tpu_resource_t resource ,
                           SgdnnTensor_t input,
                           int *dim_order,
                           SgdnnTensor_t output ,
                           bool non_blocking ) {
  SGDNN_CHECK ( input.dim == output.dim );
  SGDNN_CHECK ( input.dtype == output.dtype );
  // SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  // SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_permute_t api;
  api.dim = input.dim;
  int size = 1;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
    api.dim_order[i] = dim_order[i];
    size *= input.shape[i];
    api.stride[i] = input.stride[i];
  }
  size *= sizeof(float);
  tpu_device_mem_t dev_mem;
  tpu_status_t status = sgdnnMallocDeviceByte(resource , &dev_mem, size);

  if(SG_SUCCESS != status){
    printf("malloc device error \r\n");
    return status;
  }
  api.trans_buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  api.copy_buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  api.dtype = sgdnnTPUKernelDType( input.dtype );
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_permute", &api, sizeof( api ) , non_blocking) );

  return SG_SUCCESS;
}

tpu_status_t sgdnnTopk ( tpu_resource_t resource ,
                        SgdnnTensor_t input,
                        int k,
                        int axis,
                        bool largest,
                        bool sorted,
                        SgdnnTensor_t value,
                        SgdnnTensor_t index ,
                        bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == value.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_INT32 ||
                input.dtype == SGDNN_DTYPE_UINT32 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &value ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &index ) );

  sg_api_topk_t api;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }

  api.dim = input.dim;
  api.k = k;
  api.axis = axis;
  api.largest = largest;
  api.sorted = sorted;
  api.input_global_addr = input.addr;
  api.value_global_addr = value.addr;
  api.index_global_addr = index.addr;
  api.dtype = sgdnnTPUKernelDType( input.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_topk", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_topk_multi_core", &api, sizeof( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnReduceMaxOrMin(tpu_resource_t resource , SgdnnTensor_t input,
                                int* reduction_dim, int reduction_dim_length,
                                int keepdim, int mode, SgdnnTensor_t output,
                                bool non_blocking ) {
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_FP16 ||
              input.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
  SGDNN_CHECK(reduction_dim_length > 0 || reduction_dim_length <= input.dim);
  SGDNN_CHECK(mode == 0 || mode == 1);

#if defined BACKEND_1684X
  sg_api_reduce_max_or_min_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  tpu_device_mem_t dev_mem;
  SAFE_CALL(sgdnnMallocDeviceByte(resource , &dev_mem, sgdnnTensorBytes(&input)));
  api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  for (int i = 0; i < reduction_dim_length; i++) {
    api.reduction_dim[i] = reduction_dim[i];
  }
  api.reduction_dim_length = reduction_dim_length;
  api.dim = input.dim;
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_reduce_max_or_min",
                                 &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_reduce_max_or_min_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  tpu_device_mem_t dev_mem;
  SAFE_CALL(sgdnnMallocDeviceByte(resource , &dev_mem, sgdnnTensorBytes(&input)));
  api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  for (int i = 0; i < reduction_dim_length; i++) {
    api.reduction_dim[i] = reduction_dim[i];
  }
  api.reduction_dim_length = reduction_dim_length;
  api.dim = input.dim;
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource ,
                                 "tpu_kernel_api_reduce_max_or_min_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNonzero ( tpu_resource_t resource ,
                           SgdnnTensor_t self,
                           SgdnnTensor_t out,
                           SgdnnTensor_t num ,
                           bool non_blocking )
{
  SGDNN_CHECK ( self.dtype == SGDNN_DTYPE_INT8  ||
                self.dtype == SGDNN_DTYPE_UINT8 ||
                self.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &self ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );

  #if defined BACKEND_1684X
  sg_api_nonzero_t api;
  api.input_global_addr = self.addr;
  api.output_global_addr = out.addr;
  api.num_global_addr = num.addr;
  api.dim = self.dim;
  int size = 1;
  for ( int i = 0; i < self.dim; ++i )
  {
    api.shape[i] = self.shape[i];
    size *= self.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( self.dtype );
  size *= sizeof(int);
  tpu_device_mem_t index_dev_mem;
  tpu_status_t status = sgdnnMallocDeviceByte(resource , &index_dev_mem, size);
  if(SG_SUCCESS != status){
    printf("malloc device error \r\n");
    return status;
  }
  api.index_global_addr = sgdnnGetDeviceAddr(index_dev_mem);

  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_nonzero", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_nonzero_multi_core_t api;
  api.input_global_addr = self.addr;
  api.output_global_addr = out.addr;
  api.num_global_addr = num.addr;
  api.dim = self.dim;
  int size = 1;
  for ( int i = 0; i < self.dim; ++i )
  {
    api.shape[i] = self.shape[i];
    size *= self.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( self.dtype );
  size *= sizeof(int);
  tpu_device_mem_t index_dev_mem;
  tpu_status_t status = sgdnnMallocDeviceByte(resource, &index_dev_mem, size);
  if(SG_SUCCESS != status){
    printf("malloc device error \r\n");
    return status;
  }
  api.index_global_addr = sgdnnGetDeviceAddr(index_dev_mem);

  tpu_device_mem_t num_buffer_dev_mem;
  status = sgdnnMallocDeviceByte(resource, &num_buffer_dev_mem, 8 * 64);; // buffer need align to 64 bytes
  if(SG_SUCCESS != status){
    printf("malloc device error \r\n");
    return status;
  }
  api.num_buffer_global_addr = sgdnnGetDeviceAddr(num_buffer_dev_mem);

  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource, "tpu_kernel_api_nonzero_multicore", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnRepeat ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          int *repeat_times,
                          int repeat_dim,
                          SgdnnTensor_t output ,
                          bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dim <= output.dim );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  sg_api_repeat_t api;
  api.dim = input.dim;
  api.repeat_dim = repeat_dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  for(int i = 0 ; i < repeat_dim; ++i) {
    api.repeat_times[i] = repeat_times[i];
  }
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  if (input.dtype == SGDNN_DTYPE_INT64 ){
    api.shape[input.dim - 1] *= 2;
    api.dtype = sgdnnTPUKernelDType( SGDNN_DTYPE_INT32 );
  } else {
    api.dtype = sgdnnTPUKernelDType( input.dtype );
  }
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_repeat", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  sg_api_repeat_t api;
  api.dim = input.dim;
  api.repeat_dim = repeat_dim;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  for(int i = 0 ; i < repeat_dim; ++i) {
    api.repeat_times[i] = repeat_times[i];
  }
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dtype = sgdnnTPUKernelDType( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_repeat_multi_core", &api, sizeof( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnArg(tpu_resource_t resource , SgdnnTensor_t input, int axis,
                     int mode, SgdnnTensor_t values, SgdnnTensor_t indices,
                     bool non_blocking ) {
  // SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32);
  SGDNN_CHECK(input.dtype == values.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&values, &indices));
  SGDNN_CHECK(mode == 0 || mode == 1 || mode == 2 || mode == 3);

#if defined BACKEND_1684X
  sg_api_reduce_arg_t api;
  api.input_global_addr = input.addr;
  api.values_global_addr = values.addr;
  api.indices_global_addr = indices.addr;
  tpu_device_mem_t dev_mem;
  SAFE_CALL(
      sgdnnMallocDeviceByte(resource , &dev_mem, sgdnnTensorBytes(&indices)));
  api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  // transform input to [1, c, h, w], h is the axis
  api.shape[0] = 1;
  api.shape[1] = 1;
  api.shape[2] = 1;
  api.shape[3] = 1;
  if (axis == input.dim) {
    for (int i = 0; i < input.dim; ++i) {
      api.shape[2] *= input.shape[i];
    }
  } else {
    for (int i = 0; i < axis; i++) {
      api.shape[1] *= input.shape[i];
    }
    api.shape[2] = input.shape[axis];
    for (int i = axis + 1; i < input.dim; ++i) {
      api.shape[3] *= input.shape[i];
    }
  }
  api.axis = 2;
  api.dim = 4;
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_arg", &api, sizeof(api)));
  sgdnnFreeDevice(resource , dev_mem);
#elif defined BACKEND_SG2260
  sg_api_reduce_arg_t api;
  api.input_global_addr = input.addr;
  api.values_global_addr = values.addr;
  api.indices_global_addr = indices.addr;
  tpu_device_mem_t dev_mem;
  SAFE_CALL(
      sgdnnMallocDeviceByte(resource , &dev_mem, sgdnnTensorBytes(&indices)));
  api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  // transform input to [1, c, h, w], h is the axis
  api.shape[0] = 1;
  api.shape[1] = 1;
  api.shape[2] = 1;
  api.shape[3] = 1;
  if (axis == input.dim) {
    for (int i = 0; i < input.dim; ++i) {
      api.shape[2] *= input.shape[i];
    }
  } else {
    for (int i = 0; i < axis; i++) {
      api.shape[1] *= input.shape[i];
    }
    api.shape[2] = input.shape[axis];
    for (int i = axis + 1; i < input.dim; ++i) {
      api.shape[3] *= input.shape[i];
    }
  }
  api.axis = 2;
  api.dim = 4;
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_arg_muti_core", &api,
                                 sizeof(api), non_blocking));
  sgdnnFreeDevice(resource , dev_mem);
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnHardtanh ( tpu_resource_t resource ,
                            SgdnnTensor_t input,
                            float min_value,
                            float max_value,
                            SgdnnTensor_t output ,
                            bool non_blocking ) {
  SGDNN_CHECK ( input.dim == output.dim );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_hardtanh_t api;
  api.dim = input.dim;
  api.min_value = min_value;
  api.max_value = max_value;
  for(int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dtype = sgdnnTPUKernelDType( input.dtype );

#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_hardtanh", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_hardtanh_multi_core", &api, sizeof( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnHypot ( tpu_resource_t resource ,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &other ) );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  sg_api_hypot_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
   SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_hypot", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnHypotBcast ( tpu_resource_t resource ,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output ,
                              bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &other ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  sg_api_hypot_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.input_shape[i] = input.shape[i];
  }
  for ( int i = 0; i < other.dim; ++i ) {
    api.other_shape[i] = other.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_hypot_bcast", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnHypotC ( tpu_resource_t resource ,
                          SgdnnTensor_t input,
                          float scalar,
                          SgdnnTensor_t output ,
                          bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 );
  SGDNN_CHECK ( input.dim == output.dim );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ));
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  sg_api_hypot_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i ) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.scalar = scalar;
  SAFE_CALL( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_hypot_c", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnNextafterC(tpu_resource_t resource , float scalar,
                            SgdnnTensor_t other, SgdnnTensor_t output,
                            bool non_blocking ) {
  SGDNN_CHECK(other.dtype == SGDNN_DTYPE_FP32 ||
              other.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(other.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&other, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_nextafterc_t api;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = other.dim;
  api.scalar = scalar;
  for (int i = 0; i < other.dim; ++i) {
    api.shape[i] = other.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_nextafterc", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_nextafterc_t api;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = other.dim;
  api.scalar = scalar;
  for (int i = 0; i < other.dim; ++i) {
    api.shape[i] = other.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_nextafterc_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNextafter_C(tpu_resource_t resource , SgdnnTensor_t input,
                             float scalar, SgdnnTensor_t output,
                             bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_nextafter_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_nextafter_c", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_nextafter_c_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  api.scalar = scalar;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(
      resource , "tpu_kernel_api_nextafter_c_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNextafter(tpu_resource_t resource , SgdnnTensor_t input,
                           SgdnnTensor_t other, SgdnnTensor_t output,
                           bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &other));
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_nextafter_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_nextafter", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_nextafter_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for (int i = 0; i < input.dim; ++i) {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(output.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_nextafter_multi_core",
                                 &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnNextafterBcast(tpu_resource_t resource , SgdnnTensor_t input,
                                SgdnnTensor_t other, SgdnnTensor_t output,
                                bool non_blocking ) {
  SGDNN_CHECK(input.dtype == SGDNN_DTYPE_FP32 ||
              input.dtype == SGDNN_DTYPE_BF16);
  SGDNN_CHECK(input.dtype == other.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&other));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));
#if defined BACKEND_1684X
  sg_api_nextafter_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_nextafter_bcast", &api,
                                 sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_nextafter_bcast_t api;
  api.input_global_addr = input.addr;
  api.other_global_addr = other.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.other_dim = other.dim;
  api.output_dim = output.dim;

  for (int i = 0; i < input.dim; ++i) {
    api.input_shape[i] = input.shape[i];
  }
  for (int i = 0; i < other.dim; ++i) {
    api.other_shape[i] = other.shape[i];
  }
  for (int i = 0; i < output.dim; ++i) {
    api.output_shape[i] = output.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType(other.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(
      resource , "tpu_kernel_api_nextafter_bcast_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnReduceVar ( tpu_resource_t resource ,
                             SgdnnTensor_t input,
                             int *reduce_list,
                             int reduce_dim,
                             int correction,
                             bool keepdim,
                             SgdnnTensor_t output ,
                             bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 );
  SGDNN_CHECK ( reduce_dim <= input.dim );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  sg_api_reduce_var_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  api.output_dim = output.dim;
  api.reduce_dim = reduce_dim;

  int size = 1;
  for ( int i = 0; i < input.dim; ++i ) {
    api.input_shape[i] = input.shape[i];
    size *= input.shape[i];
  }
  size *= sizeof ( float );
  tpu_device_mem_t buffer_mem, mul_mem, sum_mem;
  tpu_status_t status = sgdnnMallocDeviceByte ( resource , &buffer_mem, size );
  if ( SG_SUCCESS != status ) {
    printf ( "malloc device error \r\n" );
    return status;
  }
  api.buffer_global_addr = sgdnnGetDeviceAddr ( buffer_mem );

  status = sgdnnMallocDeviceByte ( resource , &mul_mem, size );
  if ( SG_SUCCESS != status ) {
    printf ( "malloc device error \r\n" );
    return status;
  }
  api.mul_global_addr = sgdnnGetDeviceAddr ( mul_mem );

  status = sgdnnMallocDeviceByte ( resource , &sum_mem, size );
  if ( SG_SUCCESS != status ){
    printf ( "malloc device error \r\n" );
    return status;
  }
  api.sum_global_addr = sgdnnGetDeviceAddr ( sum_mem );

  std::map<int, int> reduce_map;
  for ( int i = 0 ; i < reduce_dim; ++i ) {
    api.reduce_list[i] = reduce_list[i];
    reduce_map[reduce_list[i]] = 1;
  }
  for ( int i = 0 ; i < input.dim; ++i ) {
    if ( reduce_map.find(i) == reduce_map.end() ) {
      api.output_shape[i] = input.shape[i];
    }
    else {
      api.output_shape[i] = 1;
    }
  }
  api.correction = correction;
  api.keepdim = keepdim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_reduce_var", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnReduceVarAll ( tpu_resource_t resource ,
                                SgdnnTensor_t input,
                                int correction,
                                bool keepdim,
                                SgdnnTensor_t output ,
                                bool non_blocking ) {
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

#if defined BACKEND_1684X
  sg_api_reduce_var_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.input_dim = input.dim;
  int size = 1;
  for ( int i = 0; i < input.dim; ++i ) {
    api.input_shape[i] = input.shape[i];
    size *= input.shape[i];
  }
  size *= sizeof ( float );
  tpu_device_mem_t buffer_mem, mul_mem;
  tpu_status_t status = sgdnnMallocDeviceByte ( resource , &buffer_mem, size );
  if ( SG_SUCCESS != status ) {
    printf ( "malloc device error \r\n" );
    return status;
  }
  api.buffer_global_addr = sgdnnGetDeviceAddr ( buffer_mem );

  status = sgdnnMallocDeviceByte ( resource , &mul_mem, size );
  if ( SG_SUCCESS != status ) {
    printf ( "malloc device error \r\n" );
    return status;
  }
  api.mul_global_addr = sgdnnGetDeviceAddr ( mul_mem );

  api.correction = correction;
  api.keepdim = keepdim;
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_reduce_var_all", &api, sizeof( api ) ) );
#elif defined BACKEND_SG2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif

  return SG_SUCCESS;
}

tpu_status_t sgdnnTriangularize ( tpu_resource_t resource ,
                      SgdnnTensor_t self,
                      int is_upper,
                      int diagonal,
                      SgdnnTensor_t out ,
                      bool non_blocking )
                      {
  SGDNN_CHECK ( self.dtype == out.dtype );
  SGDNN_CHECK ( self.dtype == SGDNN_DTYPE_FP32 ||
                self.dtype == SGDNN_DTYPE_FP16 ||
                self.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &self, &out ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &self ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &out ) );
  sg_api_triangularize_t api;
  api.input_global_addr = self.addr;
  api.output_global_addr = out.addr;
  api.dims = self.dim;
  for ( int i = 0; i < self.dim; ++i )
  {
    api.shape[i] = self.shape[i];
  }
  api.diagonal = diagonal;
  api.is_upper = is_upper;
  api.dtype = sgdnnTPUKernelDType ( self.dtype );
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_triangularize", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_triangularize_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnCbrt ( tpu_resource_t resource ,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output ,
                        bool non_blocking )
{
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );

  sg_api_cbrt_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );

#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_cbrt", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  SAFE_CALL ( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_cbrt_multi_core", &api, sizeof ( api ) , non_blocking) );
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnPad(tpu_resource_t resource , SgdnnTensor_t input, int* pad,
                     int pad_size, float value, int mode, bool pad3d,
                     SgdnnTensor_t output,
                     bool non_blocking ) {
  SGDNN_CHECK(2 * input.dim >= pad_size);
  SGDNN_CHECK(pad_size % 2 == 0);
  SGDNN_CHECK(input.dim == output.dim);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(sgdnnIsTensorContiguous(&input));
  SGDNN_CHECK(sgdnnIsTensorContiguous(&output));

#if defined BACKEND_1684X
  sg_api_pad_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.buffer_global_addr = 0;
  api.pad3d = pad3d;
  api.dim = pad3d ? 5 : 4;
  api.pad_size = pad_size;
  if (pad3d) {
    tpu_device_mem_t dev_mem;
    SAFE_CALL(
        sgdnnMallocDeviceByte(resource , &dev_mem, sgdnnTensorBytes(&output)));
    api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  }
  for (int i = 0; i < api.dim; ++i) {
    if (i < api.dim - input.dim) {
      api.shape[i] = 1;
    } else {
      api.shape[i] = input.shape[i - (api.dim - input.dim)];
    }
  }
  for (int i = pad_size / 2 - 1, j = 0; i >= 0; --i, j += 2) {
    api.pad[2 * i] = pad[j];
    api.pad[2 * i + 1] = pad[j + 1];
  }
  api.value = value;
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_pad", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_pad_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.buffer_global_addr = 0;
  api.pad3d = pad3d;
  api.dim = pad3d ? 5 : 4;
  api.pad_size = pad_size;
  if (pad3d) {
    tpu_device_mem_t dev_mem;
    SAFE_CALL(
        sgdnnMallocDeviceByte(resource , &dev_mem, sgdnnTensorBytes(&output)));
    api.buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);
  }
  for (int i = 0; i < api.dim; ++i) {
    if (i < api.dim - input.dim) {
      api.shape[i] = 1;
    } else {
      api.shape[i] = input.shape[i - (api.dim - input.dim)];
    }
  }
  for (int i = pad_size / 2 - 1, j = 0; i >= 0; --i, j += 2) {
    api.pad[2 * i] = pad[j];
    api.pad[2 * i + 1] = pad[j + 1];
  }
  api.value = value;
  api.mode = mode;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_pad_multi_core", &api,
                                 sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnSliceScatter(tpu_resource_t resource , SgdnnTensor_t input,
                              SgdnnTensor_t src, SgdnnTensor_t indices, int dim,
                              SgdnnTensor_t output,
                              bool non_blocking ) {
  SGDNN_CHECK(
      input.dtype == SGDNN_DTYPE_FP32 || input.dtype == SGDNN_DTYPE_FP16 ||
      input.dtype == SGDNN_DTYPE_BF16 || input.dtype == SGDNN_DTYPE_INT32);
  SGDNN_CHECK(input.dtype == src.dtype);
  SGDNN_CHECK(input.dtype == output.dtype);
  SGDNN_CHECK(input.dim == output.dim);
  SGDNN_CHECK(input.dim == src.dim);
  SGDNN_CHECK(sgdnnIsSameShape(&input, &output));

#if defined BACKEND_1684X
  sg_api_slice_scatter_t api;
  api.input_global_addr = input.addr;
  api.src_global_addr = src.addr;
  api.output_global_addr = output.addr;
  api.indices_global_addr = indices.addr;

  api.input_dim = input.dim;

  for (int i = 0; i < input.dim; i++){
    api.input_shape[i] = input.shape[i];
    api.src_shape[i] = src.shape[i];
  }
  api.dim = dim;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunch(resource , "tpu_kernel_api_slice_scatter", &api, sizeof(api)));
#elif defined BACKEND_SG2260
  sg_api_slice_scatter_t api;
  api.input_global_addr = input.addr;
  api.src_global_addr = src.addr;
  api.output_global_addr = output.addr;
  api.indices_global_addr = indices.addr;

  api.input_dim = input.dim;

  for (int i = 0; i < input.dim; i++){
    api.input_shape[i] = input.shape[i];
    api.src_shape[i] = src.shape[i];
  }
  api.dim = dim;
  api.dtype = sgdnnTPUKernelDType(input.dtype);
  SAFE_CALL(
      sgdnnTPUKernelLaunchMultiCore(resource , "tpu_kernel_api_slice_scatter_multi_core", &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}

bool _sgdnnInfCheckAndUnscale( tpu_resource_t resource ,
                              SgdnnTensor_t& input,
                              SgdnnTensor_t& found_inf,
                              float inv_scale,
                              bool non_blocking )
{
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  sg_api_inf_check_unscale_t api;
  api.input_global_addr = input.addr;
  api.found_inf_global_addr = found_inf.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; i++ ) { api.shape[i] = input.shape[i]; }
  api.inv_scale = inv_scale;
  api.idtype = sgdnnTPUKernelDType( input.dtype );
  api.found_inf_dtype = sgdnnTPUKernelDType( found_inf.dtype );
  SAFE_CALL( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_inf_check_and_unscale", &api, sizeof(api) , non_blocking) );

  tpu_device_mem_t SrcMem = sgdnnMemFromDevice ( found_inf.addr, sgdnnDataSize(found_inf.dtype) );
  float found = 0;
  SAFE_CALL (sgdnnMemcpyD2S ( resource , &found, SrcMem , sgdnnDataSize(found_inf.dtype)));
  return found == 0.0;
}

#if defined BACKEND_SG2260
bool _sgdnnInfCheckAndUnscale_multi_core( tpu_resource_t resource ,
                              SgdnnTensor_t& input,
                              SgdnnTensor_t& found_inf,
                              float inv_scale,
                              bool non_blocking )
{
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  sg_api_inf_check_unscale_multi_core_t api;
  api.input_global_addr = input.addr;
  api.found_inf_global_addr = found_inf.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; i++ ) { api.shape[i] = input.shape[i]; }
  api.inv_scale = inv_scale;
  api.idtype = sgdnnTPUKernelDType( input.dtype );
  api.found_inf_dtype = sgdnnTPUKernelDType( found_inf.dtype );
  // a buffer to store middle results
  unsigned int size = 8 * 64; // buffer need align to 64 bytes
  tpu_device_mem_t dev_mem;
  tpu_status_t err = sgdnnMallocDeviceByte(resource , &dev_mem, size);
  if (SG_SUCCESS != err){
    printf("malloc device error \r\n");
    return err;
  }
  api.found_inf_buffer_global_addr = sgdnnGetDeviceAddr(dev_mem);

  SAFE_CALL( sgdnnTPUKernelLaunchMultiCore ( resource , "tpu_kernel_api_inf_check_and_unscale_multi_core", &api, sizeof(api) , non_blocking) );

  tpu_device_mem_t SrcMem = sgdnnMemFromDevice ( found_inf.addr, sgdnnDataSize(found_inf.dtype) );
  float found = 0;
  SAFE_CALL (sgdnnMemcpyD2S ( resource , &found, SrcMem, sgdnnDataSize(found_inf.dtype)));
  return found == 0.0;
}
#endif

tpu_status_t sgdnnInfCheckAndUnscale( tpu_resource_t resource ,
                                    std::vector<SgdnnTensor_t>& inputs,
                                    SgdnnTensor_t found_inf,
                                    float inv_scale ,
                                    bool non_blocking )
{
#if defined BACKEND_1684X
  for (auto& input : inputs){
    if (!_sgdnnInfCheckAndUnscale(resource , input, found_inf, inv_scale, non_blocking)){
      break;
    }
  }
#elif defined BACKEND_SG2260
  for (auto& input : inputs){
    if (!_sgdnnInfCheckAndUnscale_multi_core(resource , input, found_inf, inv_scale, non_blocking)){
      break;
    }
  }
#else
  SGDNN_CHECK(false);
#endif
  return SG_SUCCESS;
}
