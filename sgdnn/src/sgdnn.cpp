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
static TPUKernelLauncher* pkernel_launcher = nullptr;
tpuRtKernelModule_t get_kernel_module(tpuRtStream_t stream)
{
  return pkernel_launcher->get_kernel_module(stream);
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
  if (pkernel_launcher == nullptr)
  {
    pkernel_launcher = new TPUKernelLauncher();
  }
  SGDNN_CHECK( pkernel_launcher->register_kernel_module(resource) == SG_SUCCESS);
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
  if ( pkernel_launcher )
  {
    SGDNN_CHECK ( pkernel_launcher->unload_kernel_module(resource) == SG_SUCCESS );
    delete pkernel_launcher;
    pkernel_launcher = nullptr;
  }
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
    return pkernel_launcher->launch_async( func_name, api, api_size, resource, group_num, block_num );
  else
    return pkernel_launcher->launch_sync( func_name, api, api_size, resource, group_num, block_num );
#else
  SGDNN_CHECK ( false );
#endif
}

#if defined BACKEND_SG2260
tpu_status_t sgdnnCacheMalloc(void** dev_ptr, int64_t size)
{
  return pkernel_launcher->cache_malloc(dev_ptr, size);
}

tpu_status_t sgdnnCacheFree( void* dev_ptr, tpu_resource_t resource )
{
  return pkernel_launcher->cache_free(dev_ptr, resource);
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
    // SGDNN_CHECK ( input.shape[0] == output.shape[0] );
    // SGDNN_CHECK ( input.shape[1] == output.shape[2] * output.shape[3] );
    // SGDNN_CHECK ( input.shape[2] == DIV_UP ( output.shape[1], 32 ) );
    // SGDNN_CHECK ( input.shape[3] == 32 );
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
  sg_api_conv_weight_recover_t api;
  api.input_global_addr = input.addr;
  api.output_global_addr = output.addr;
  for ( int i = 0; i < output.dim; ++i )
  {
    api.shape[i] = output.shape[i];
  }
  api.mode = mode;
#if defined BACKEND_1684X
  SAFE_CALL ( sgdnnTPUKernelLaunch ( resource , "tpu_kernel_api_conv_weight_recover", &api, sizeof ( api ) ) );
#elif defined BACKEND_SG2260
  // SGDNN_CHECK ( false );
  SAFE_CALL( sgdnnTPUKernelLaunchMultiCore( resource, "tpu_kernel_api_conv_weight_recover_multi_core",  &api, sizeof(api), non_blocking));
#else
  SGDNN_CHECK ( false );
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
