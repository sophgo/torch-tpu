#include "sgdnn_api.h"
#include "sg_api_struct.h"
#include "kernel_module_data.h"
#include <map>
#include <memory>
#include <stdio.h>

#define SGDNN_BACKEND_1684X

#define SGDNN_CHECK(expression) \
do \
{ \
  if ( !( expression ) ) \
  { \
    printf ( "%s:%d:%s: %s failed\n", __FILE__, __LINE__, __func__, #expression ); \
    throw; \
  } \
} \
while ( false )

#define SAFE_CALL(cmd) \
SGDNN_CHECK ( ( cmd ) == BM_SUCCESS )

#define DIV_UP(a, b) ( ( ( a ) + ( b ) - 1 ) / ( b ) )

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

static inline void sgdnn32ICShape ( const int * shape, int * _32ic_shape )
{
  _32ic_shape[0] = shape[0];
  _32ic_shape[1] = shape[2] * shape[3];
  _32ic_shape[2] = DIV_UP ( shape[1], 32 );
  _32ic_shape[3] = 32;
}

static inline void sgdnn32OCShape ( const int * shape, int * _32oc_shape )
{
  _32oc_shape[0] = shape[1];
  _32oc_shape[1] = shape[2] * shape[3];
  _32oc_shape[2] = DIV_UP ( shape[0], 32 );
  _32oc_shape[3] = 32;
}

static inline void sgdnnContiguousStride ( const int * shape, int dim,  int * stride )
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

static inline int sgdnnTPUKernelDType ( SgdnnDataType_t dtype )
{
#if defined SGDNN_BACKEND_1684X
  if ( dtype == SGDNN_DTYPE_INT8 )        { return ( 0 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT8 )  { return ( 0 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_INT16 )  { return ( 3 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT16 ) { return ( 3 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_FP16 )   { return ( 1 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_BF16 )   { return ( 5 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_INT32 )  { return ( 4 << 1 ) | 1; }
  else if ( dtype == SGDNN_DTYPE_UINT32 ) { return ( 4 << 1 ) | 0; }
  else if ( dtype == SGDNN_DTYPE_FP32 )   { return ( 2 << 1 ) | 1; }
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  printf ( "Data type %d is not supported by the current backend\n", dtype );
  SGDNN_CHECK ( false );
}

static std::map<bm_handle_t, tpu_kernel_module_t> tpu_kernel_module;

bm_status_t sgdnnInitialize ( bm_handle_t handle )
{
#if defined SGDNN_BACKEND_1684X
  if ( tpu_kernel_module.find ( handle ) != tpu_kernel_module.end() )
  {
    return BM_SUCCESS;
  }
  const unsigned int * p = kernel_module_data;
  size_t length = sizeof ( kernel_module_data );
  tpu_kernel_module_t tpu_module = tpu_kernel_load_module ( handle, ( const char * ) p, length );
  tpu_kernel_module.insert ( std::pair<bm_handle_t, tpu_kernel_module_t> ( handle, tpu_module ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnDeinitialize ( bm_handle_t handle )
{
#if defined SGDNN_BACKEND_1684X
  if ( tpu_kernel_module.find ( handle ) == tpu_kernel_module.end() )
  {
    return BM_SUCCESS;
  }
  SGDNN_CHECK ( tpu_kernel_module.erase ( handle ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

static bm_status_t sgdnnTPUKernelLaunch (
bm_handle_t handle,
const char * func_name,
const void * api,
size_t api_size )
{
#if defined SGDNN_BACKEND_1684X
  tpu_kernel_function_t func_id;
  tpu_kernel_module_t tpu_module = tpu_kernel_module[handle];
  func_id = tpu_kernel_get_function ( handle, tpu_module, func_name );
  return tpu_kernel_launch ( handle, func_id, ( void * ) api, api_size );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
}

bm_status_t sgdnnConv2d ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight,
                          SgdnnTensor_t bias,
                          SgdnnConv2dParam_t param,
                          SgdnnTensor_t output )
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
  SGDNN_CHECK ( weight.dim == 4 );
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
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &bias ) );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  bm_device_mem_t weight_reordered_mem;
  if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
  {
    SgdnnTensor_t weight_reordered;
    weight_reordered.dim = 4;
    sgdnn32ICShape ( weight.shape, weight_reordered.shape );
    sgdnnContiguousStride ( weight_reordered.shape, 4, weight_reordered.stride );
    weight_reordered.dtype = weight.dtype;
    SAFE_CALL ( bm_malloc_device_byte ( handle, &weight_reordered_mem, sgdnnTensorBytes ( &weight_reordered ) ) );
    weight_reordered.addr = bm_mem_get_device_addr ( weight_reordered_mem );
    SAFE_CALL ( sgdnnReorderConv2dWeight ( handle, weight, 0, weight_reordered ) );
  }
  sg_api_conv2d_t api;
  api.input_global_addr = input.addr;
  if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
  {
    api.weight_global_addr = bm_mem_get_device_addr ( weight_reordered_mem );
  }
  else
  {
    api.weight_global_addr = weight.addr;
  }
  api.bias_global_addr = bias.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.input_shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.groups = param.groups;
  api.output_c = output.shape[1];
  api.kernel[0] = weight.shape[2];
  api.kernel[1] = weight.shape[3];
  api.stride[0] = param.stride_h;
  api.stride[1] = param.stride_w;
  api.dilation[0] = param.dilation_h;
  api.dilation[1] = param.dilation_w;
  api.pad[0] = param.pad_h;
  api.pad[1] = param.pad_h;
  api.pad[2] = param.pad_w;
  api.pad[3] = param.pad_w;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_conv2d", &api, sizeof ( api ) ) );
  if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
  {
    bm_free_device ( handle, weight_reordered_mem );
  }
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnConv2dBackward ( bm_handle_t handle,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t weight,
                                  SgdnnConv2dParam_t param,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_weight,
                                  SgdnnTensor_t grad_bias )
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
  SGDNN_CHECK ( weight.dim == 4 );
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
  SGDNN_CHECK ( weight.shape[0] = grad_output.shape[1] );
  SGDNN_CHECK ( weight.shape[1] = input.shape[1] / param.groups );
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
    SGDNN_CHECK ( sgdnnIsSameShape ( &weight, &grad_weight ) );
  }
  // TODO: CHECK SHAPE
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &weight ) );
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
#if defined SGDNN_BACKEND_1684X
  if ( grad_input.addr != 0 || grad_weight.addr != 0 || grad_bias.addr != 0 )
  {
    bm_device_mem_t buffer_mem;
    if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
    {
      SgdnnTensor_t weight_reordered;
      weight_reordered.dim = 4;
      sgdnn32OCShape ( weight.shape, weight_reordered.shape );
      weight_reordered.dtype = weight.dtype;
      size_t weight_reordered_size = sgdnnTensorBytes ( &weight_reordered );
      SgdnnTensor_t grad_output_reordered;
      sgdnn32OCShape ( grad_output.shape, grad_output_reordered.shape );
      grad_output_reordered.dtype = grad_output.dtype;
      size_t grad_output_reordered_size = sgdnnTensorBytes ( &grad_output_reordered );
      size_t buffer_size = weight_reordered_size > grad_output_reordered_size ? weight_reordered_size : grad_output_reordered_size;
      SAFE_CALL ( bm_malloc_device_byte ( handle, &buffer_mem, buffer_size ) );
    }
    sg_api_conv2d_backward_t api;
    api.input_global_addr = input.addr;
    api.weight_global_addr = weight.addr;
    api.grad_output_global_addr = grad_output.addr;
    api.grad_input_global_addr = grad_input.addr;
    api.grad_weight_global_addr = grad_weight.addr;
    api.grad_bias_global_addr = grad_bias.addr;
    api.buffer_global_addr = weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 ?
                             bm_mem_get_device_addr ( buffer_mem ) : 0;
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
    api.kernel[0] = weight.shape[2];
    api.kernel[1] = weight.shape[3];
    api.stride[0] = param.stride_h;
    api.stride[1] = param.stride_w;
    api.dilation[0] = param.dilation_h;
    api.dilation[1] = param.dilation_w;
    api.pad[0] = param.pad_h;
    api.pad[1] = param.pad_h;
    api.pad[2] = param.pad_w;
    api.pad[3] = param.pad_w;
    SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_conv2d_backward", &api, sizeof ( api ) ) );
    if ( weight.dtype == SGDNN_DTYPE_FP16 || weight.dtype == SGDNN_DTYPE_BF16 )
    {
      bm_free_device ( handle, buffer_mem );
    }
  }
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnBatchnorm2d ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t weight,
                               SgdnnTensor_t bias,
                               float eps,
                               SgdnnTensor_t running_mean,
                               SgdnnTensor_t running_var,
                               float momentum,
                               SgdnnTensor_t output,
                               SgdnnTensor_t saved_mean,
                               SgdnnTensor_t saved_invstd )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_batchnorm2d", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnBatchnorm2dBackward ( bm_handle_t handle,
                                       SgdnnTensor_t grad_output,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t weight,
                                       SgdnnTensor_t saved_mean,
                                       SgdnnTensor_t saved_invstd,
                                       SgdnnTensor_t grad_input,
                                       SgdnnTensor_t grad_weight,
                                       SgdnnTensor_t grad_bias )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_batchnorm2d_backward", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnLayernorm ( bm_handle_t handle,
                             SgdnnTensor_t input,
                             SgdnnTensor_t weight,
                             SgdnnTensor_t bias,
                             int start_dim,
                             float eps,
                             SgdnnTensor_t output,
                             SgdnnTensor_t mean,
                             SgdnnTensor_t rstd )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_layernorm", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnLayernormBackward ( bm_handle_t handle,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t input,
                                     SgdnnTensor_t weight,
                                     SgdnnTensor_t mean,
                                     SgdnnTensor_t rstd,
                                     int start_dim,
                                     SgdnnTensor_t grad_input,
                                     SgdnnTensor_t grad_weight,
                                     SgdnnTensor_t grad_bias )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_layernorm_backward", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnReorderConv2dWeight ( bm_handle_t handle,
                                       SgdnnTensor_t input,
                                       int mode,
                                       SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( input.dim == 4 );
  SGDNN_CHECK ( output.dim == 4 );
  SGDNN_CHECK ( mode == 0 || mode == 1 );
  if ( mode == 0 )
  {
    SGDNN_CHECK ( output.shape[0] == input.shape[0] );
    SGDNN_CHECK ( output.shape[1] == input.shape[2] * input.shape[3] );
    SGDNN_CHECK ( output.shape[2] == DIV_UP ( input.shape[1], 32 ) );
    SGDNN_CHECK ( output.shape[3] == 32 );
  }
  else if ( mode == 1 )
  {
    SGDNN_CHECK ( output.shape[0] == input.shape[1] );
    SGDNN_CHECK ( output.shape[1] == input.shape[2] * input.shape[3] );
    SGDNN_CHECK ( output.shape[2] == DIV_UP ( input.shape[0], 32 ) );
    SGDNN_CHECK ( output.shape[3] == 32 );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_conv_weight_reorder_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.mode = mode;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_conv_weight_reorder", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnSoftmax ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           int dim,
                           SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_softmax", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnSoftmaxBackward ( bm_handle_t handle,
                                   SgdnnTensor_t grad_output,
                                   SgdnnTensor_t output,
                                   int dim,
                                   SgdnnTensor_t grad_input )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_softmax_backward", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnReduce ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          int start_dim,
                          int end_dim,
                          int keepdim,
                          int mode,
                          SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_reduce", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnStridedCopy ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_strided_copy", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnConvert ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           SgdnnTensor_t output )
{
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_dtype_convert", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnWhere ( bm_handle_t handle,
                         SgdnnTensor_t cond,
                         SgdnnTensor_t self,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_where", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnConcat ( bm_handle_t handle,
                          const SgdnnTensor_t * inputs,
                          int input_num,
                          int dim,
                          SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_concat", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnIndexSelect ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t indices,
                               int dim,
                               SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
  SGDNN_CHECK ( indices.dtype == SGDNN_DTYPE_INT32 );
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_index_select", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnFill ( bm_handle_t handle,
                        const void * scalar_ptr,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_const_fill", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnAdd ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       float scalar,
                       SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
  sg_api_add_t api;
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_add", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnAddBcast ( bm_handle_t handle,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            float scalar,
                            SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == other.dtype );
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
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
#if defined SGDNN_BACKEND_1684X
  sg_api_bcast_add_t api;
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_bcast_add", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnSub ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       float scalar,
                       SgdnnTensor_t output )
{
  SGDNN_CHECK ( false );
}

bm_status_t sgdnnMul ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output )
{
  SGDNN_CHECK ( false );
}

bm_status_t sgdnnDiv ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output )
{
  SGDNN_CHECK ( false );
}

bm_status_t sgdnnAddC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_addc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_addc", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnMulC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_mulc_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_mulc", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnCSub ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_csub_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_csub", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnCDiv ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_cdiv_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  api.value = scalar;
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_cdiv", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnCrossEntropyLoss ( bm_handle_t handle,
                                    SgdnnTensor_t input,
                                    SgdnnTensor_t target,
                                    int reduction,
                                    float label_smoothing,
                                    SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_cross_entropy_loss", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnCrossEntropyLossBackward (
bm_handle_t handle,
SgdnnTensor_t input,
SgdnnTensor_t target,
SgdnnTensor_t grad_output,
int reduction,
float label_smoothing,
SgdnnTensor_t grad_input )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_cross_entropy_loss_backward", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnMatmul ( bm_handle_t handle,
                          SgdnnTensor_t left,
                          SgdnnTensor_t right,
                          SgdnnTensor_t bias,
                          SgdnnTensor_t output )
{
  SGDNN_CHECK ( left.dtype == right.dtype );
  SGDNN_CHECK ( left.dtype == output.dtype );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( left.dtype == bias.dtype );
  }
  SGDNN_CHECK ( left.dtype == SGDNN_DTYPE_FP32 ||
                left.dtype == SGDNN_DTYPE_FP16 ||
                left.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( left.dim == 2 );
  SGDNN_CHECK ( right.dim == 2 );
  SGDNN_CHECK ( output.dim == 2 );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.dim == 1 );
  }
  SGDNN_CHECK ( left.shape[0] == output.shape[0] );
  SGDNN_CHECK ( right.shape[1] == output.shape[1] );
  SGDNN_CHECK ( left.shape[1] == right.shape[0] );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( bias.shape[0] == output.shape[1] );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &left ) || sgdnnIsTensorTransposed ( &left ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &right ) || sgdnnIsTensorTransposed ( &right ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
  if ( bias.addr != 0 )
  {
    SGDNN_CHECK ( sgdnnIsTensorContiguous ( &bias ) );
  }
#if defined SGDNN_BACKEND_1684X
  bm_device_mem_t left_contiguous_mem;
  if ( sgdnnIsTensorTransposed ( &left ) )
  {
    SgdnnTensor_t left_contiguous;
    left_contiguous.dim = 2;
    left_contiguous.shape[0] = left.shape[0];
    left_contiguous.shape[1] = left.shape[1];
    left_contiguous.stride[0] = left.shape[1];
    left_contiguous.stride[1] = 1;
    left_contiguous.dtype = left.dtype;
    SAFE_CALL ( bm_malloc_device_byte ( handle, &left_contiguous_mem, sgdnnTensorBytes ( &left_contiguous ) ) );
    left_contiguous.addr = bm_mem_get_device_addr ( left_contiguous_mem );
    SAFE_CALL ( sgdnnStridedCopy ( handle, left, left_contiguous ) );
  }
  sg_api_batch_matmul_t api;
  api.left_global_addr = sgdnnIsTensorTransposed ( &left ) ? bm_mem_get_device_addr ( left_contiguous_mem ) : left.addr;
  api.right_global_addr = right.addr;
  api.bias_global_addr = bias.addr;
  api.output_global_addr = output.addr;
  api.batch = 1;
  api.left_row = left.shape[0];
  api.left_column = left.shape[1];
  api.right_column = right.shape[1];
  api.is_left_transposed = 0; // sgdnnIsTensorTransposed ( &left ); // bm1684x does not support TN Matmul
  api.is_right_transposed = sgdnnIsTensorTransposed ( &right );
  api.dtype = sgdnnTPUKernelDType ( left.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_batch_matmul", &api, sizeof ( api ) ) );
  if ( sgdnnIsTensorTransposed ( &left ) )
  {
    bm_free_device ( handle, left_contiguous_mem );
  }
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnBatchMatmul ( bm_handle_t handle,
                               SgdnnTensor_t left,
                               SgdnnTensor_t right,
                               SgdnnTensor_t output )
{
  SGDNN_CHECK ( left.dtype == right.dtype );
  SGDNN_CHECK ( left.dtype == output.dtype );
  SGDNN_CHECK ( left.dtype == SGDNN_DTYPE_FP32 ||
                left.dtype == SGDNN_DTYPE_FP16 ||
                left.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( left.dim == 3 );
  SGDNN_CHECK ( right.dim == 3 );
  SGDNN_CHECK ( output.dim == 3 );
  SGDNN_CHECK ( left.shape[0] == output.shape[0] );
  SGDNN_CHECK ( left.shape[0] == output.shape[0] );
  SGDNN_CHECK ( left.shape[1] == output.shape[1] );
  SGDNN_CHECK ( right.shape[2] == output.shape[2] );
  SGDNN_CHECK ( left.shape[2] == right.shape[1] );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &left ) || sgdnnIsTensorTransposed ( &left ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &right ) || sgdnnIsTensorTransposed ( &right ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  bm_device_mem_t left_contiguous_mem;
  if ( sgdnnIsTensorTransposed ( &left ) )
  {
    SgdnnTensor_t left_contiguous;
    left_contiguous.dim = 3;
    left_contiguous.shape[0] = left.shape[0];
    left_contiguous.shape[1] = left.shape[1];
    left_contiguous.shape[2] = left.shape[2];
    left_contiguous.stride[0] = left.shape[0];
    left_contiguous.stride[1] = left.shape[2];
    left_contiguous.stride[2] = 1;
    left_contiguous.dtype = left.dtype;
    SAFE_CALL ( bm_malloc_device_byte ( handle, &left_contiguous_mem, sgdnnTensorBytes ( &left_contiguous ) ) );
    left_contiguous.addr = bm_mem_get_device_addr ( left_contiguous_mem );
    SAFE_CALL ( sgdnnStridedCopy ( handle, left, left_contiguous ) );
  }
  sg_api_batch_matmul_t api;
  api.left_global_addr = sgdnnIsTensorTransposed ( &left ) ? bm_mem_get_device_addr ( left_contiguous_mem ) : left.addr;
  api.right_global_addr = right.addr;
  api.bias_global_addr = 0;
  api.output_global_addr = output.addr;
  api.batch = left.shape[0];
  api.left_row = left.shape[1];
  api.left_column = left.shape[2];
  api.right_column = right.shape[2];
  api.is_left_transposed = 0; // sgdnnIsTensorTransposed ( &left ); // bm1684x does not support TN Matmul
  api.is_right_transposed = sgdnnIsTensorTransposed ( &right );
  api.dtype = sgdnnTPUKernelDType ( left.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_batch_matmul", &api, sizeof ( api ) ) );
  if ( sgdnnIsTensorTransposed ( &left ) )
  {
    bm_free_device ( handle, left_contiguous_mem );
  }
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnEmbeddingBackward ( bm_handle_t handle,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t indices,
                                     SgdnnTensor_t grad_input )
{
  SGDNN_CHECK ( grad_output.dtype == grad_input.dtype );
  SGDNN_CHECK ( grad_output.dtype == SGDNN_DTYPE_FP32 ||
                grad_output.dtype == SGDNN_DTYPE_FP16 ||
                grad_output.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( indices.dtype == SGDNN_DTYPE_INT32 || indices.dtype == SGDNN_DTYPE_INT64 );
  SGDNN_CHECK ( grad_input.dim == 2 );
  SGDNN_CHECK ( grad_output.shape[grad_output.dim - 1] == grad_input.shape[1] );
#if defined SGDNN_BACKEND_1684X
  sg_api_embedding_backward_t api;
  const int window_size = 64;
  SGDNN_CHECK ( window_size % 64 == 0 );
  int indices_num = 1;
  for ( int i = 0; i < indices.dim; i++ )
  {
    indices_num *= indices.shape[i];
  }
  bm_device_mem_t sorted_index, sorted_index_index, from_index, to_index;
  SAFE_CALL ( bm_malloc_device_byte ( handle, &sorted_index, indices_num * sizeof ( int ) ) );
  SAFE_CALL ( bm_malloc_device_byte ( handle, &sorted_index_index, indices_num * sizeof ( int ) ) );
  SAFE_CALL ( bm_malloc_device_byte ( handle, &from_index, window_size * sizeof ( int ) ) );
  SAFE_CALL ( bm_malloc_device_byte ( handle, &to_index, window_size * sizeof ( int ) ) );
  api.grad_output_global_addr = grad_output.addr;
  api.index_global_addr = indices.addr;
  api.grad_input_global_addr = grad_input.addr;
  api.sorted_index_global_addr = bm_mem_get_device_addr ( sorted_index );
  api.sorted_index_index_global_addr = bm_mem_get_device_addr ( sorted_index_index );
  api.from_index_global_addr = bm_mem_get_device_addr ( from_index );
  api.to_index_global_addr = bm_mem_get_device_addr ( to_index );
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_embedding_backward", &api, sizeof ( api ) ) );
  bm_free_device ( handle, sorted_index );
  bm_free_device ( handle, sorted_index_index );
  bm_free_device ( handle, from_index );
  bm_free_device ( handle, to_index );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnNorm2 ( bm_handle_t handle,
                         SgdnnTensor_t input,
                         int keepdim,
                         SgdnnTensor_t output )
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
    SGDNN_CHECK ( output.dim == 0 );
  }
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_norm2_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_norm2", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnReLU ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_relu_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_relu", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnGELU ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_gelu_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_gelu", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnReLUBackward ( bm_handle_t handle,
                                SgdnnTensor_t grad_output,
                                SgdnnTensor_t input,
                                SgdnnTensor_t grad_input )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_relu_backward", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnGELUBackward ( bm_handle_t handle,
                                SgdnnTensor_t grad_output,
                                SgdnnTensor_t input,
                                SgdnnTensor_t grad_input )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_gelu_backward", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnSqrt ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output )
{
  SGDNN_CHECK ( input.dtype == output.dtype );
  SGDNN_CHECK ( input.dtype == SGDNN_DTYPE_FP32 ||
                input.dtype == SGDNN_DTYPE_FP16 ||
                input.dtype == SGDNN_DTYPE_BF16 );
  SGDNN_CHECK ( sgdnnIsSameShape ( &input, &output ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &input ) );
  SGDNN_CHECK ( sgdnnIsTensorContiguous ( &output ) );
#if defined SGDNN_BACKEND_1684X
  sg_api_sqrt_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  api.dim = input.dim;
  for ( int i = 0; i < input.dim; ++i )
  {
    api.shape[i] = input.shape[i];
  }
  api.dtype = sgdnnTPUKernelDType ( input.dtype );
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_sqrt", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnAddCMul ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_addcmul", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}

bm_status_t sgdnnAddCDiv ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output )
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
#if defined SGDNN_BACKEND_1684X
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
  SAFE_CALL ( sgdnnTPUKernelLaunch ( handle, "tpu_kernel_api_addcdiv", &api, sizeof ( api ) ) );
#elif defined SGDNN_BACKEND_2260
  SGDNN_CHECK ( false );
#else
  SGDNN_CHECK ( false );
#endif
  return BM_SUCCESS;
}
