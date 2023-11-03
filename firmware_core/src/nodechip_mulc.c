#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

/*
 * output = input op value
 */

static inline void nodechip_opc (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
scalar_t value,
int length,
data_type_t dtype,
int op ) // 0: add, 1: mul
{
  const int dsize = tpu_data_type_size ( dtype );
  int wmax = DIV_UP ( length, NPU_NUM );
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = tpu_aligned_feature_size ( 1, wmax, dtype );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += size;
    output_local_addrs[1] = next; next += size;
    if ( ( int ) next <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( wmax > 1 )
      {
        wmax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  int todo = length;
  int done = 0;
  dim4 shape = { .n = 1, .h = 1 };
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  while ( todo != 0 )
  {
    if ( todo > NPU_NUM )
    {
      shape.c = NPU_NUM;
      shape.w = MIN ( todo / NPU_NUM, wmax );
    }
    else
    {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dsize, &shape, NULL, NULL, dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    if ( op == 0 )
    {
      if ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 )
      {
        tpu_bdc_fp_add_C ( output_local_addrs[index], input_local_addrs[index], value, &shape, NULL, NULL, dtype );
      }
      else
      {
        tpu_bdc_int_add_C ( output_local_addrs[index], input_local_addrs[index], value, &shape, NULL, NULL, dtype, dtype, dtype, 0, NO_USE, false );
      }
    }
    else if ( op == 1 )
    {
      if ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 )
      {
        tpu_bdc_fp_mul_C ( output_local_addrs[index], input_local_addrs[index], value, &shape, NULL, NULL, dtype );
      }
      else
      {
        tpu_bdc_int_mul_C ( output_local_addrs[index], input_local_addrs[index], value, &shape, NULL, NULL, dtype, dtype, dtype, 0, NO_USE, false );
      }
    }
    l2s = true;
    l2s_global_addr = output_global_addr + done * dsize;
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s )
  {
    tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
  }
}

void tpu_kernel_api_mulc ( const void * args )
{
  sg_api_mulc_t * api = ( sg_api_mulc_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 || api->dtype == DT_INT32 );
  scalar_t value;
  if ( api->dtype == DT_FP32 )
  {
    value.f32 = api->value;
  }
  else if (api->dtype == DT_INT32) {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_to_int_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  else
  {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_opc ( api->input_global_addr, api->output_global_addr, value, length, ( data_type_t ) api->dtype, 1 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_mulc );

void tpu_kernel_api_addc ( const void * args )
{
  sg_api_addc_t * api = ( sg_api_addc_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 || api->dtype == DT_INT32);
  scalar_t value;
  if ( api->dtype == DT_FP32 )
  {
    value.f32 = api->value;
  }
  else if (api->dtype == DT_INT32) {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_to_int_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  else
  {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_opc ( api->input_global_addr, api->output_global_addr, value, length, ( data_type_t ) api->dtype, 0 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_addc );

static inline void nodechip_cop (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
scalar_t value,
int length,
data_type_t dtype,
int op ) // 0: sub, 1: div
{
  const int dsize = tpu_data_type_size ( dtype );
  int wmax = DIV_UP ( length, NPU_NUM );
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t input_fp32_local_addr = 0;
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = tpu_aligned_feature_size ( 1, wmax, dtype );
    int size_fp32 = tpu_aligned_feature_size ( 1, wmax, DT_FP32 );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += size;
    output_local_addrs[1] = next; next += size;
    if ( op == 1 && dtype != DT_FP32 )
    {
      input_fp32_local_addr = next; next += size_fp32;
    }
    if ( ( int ) next <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( wmax > 1 )
      {
        wmax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  int todo = length;
  int done = 0;
  dim4 shape = { .n = 1, .h = 1 };
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  while ( todo != 0 )
  {
    if ( todo > NPU_NUM )
    {
      shape.c = NPU_NUM;
      shape.w = MIN ( todo / NPU_NUM, wmax );
    }
    else
    {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dsize, &shape, NULL, NULL, dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    if ( op == 0 )
    {
      if ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 )
      {
        tpu_bdc_fp_C_sub ( output_local_addrs[index], input_local_addrs[index], value, &shape, NULL, NULL, dtype );
      }
      else
      {
        tpu_bdc_int_C_sub ( output_local_addrs[index], input_local_addrs[index], value, &shape, NULL, NULL, dtype, dtype, dtype, 0, NO_USE, false );
      }
    }
    else if ( op == 1 )
    {
      if ( dtype == DT_FP32 )
      {
        tpu_bdc_fp32_C_div ( output_local_addrs[index], input_local_addrs[index], value.f32, &shape, NULL, NULL );
      }
      else
      {
        tpu_bdc_cast ( input_fp32_local_addr, input_local_addrs[index], &shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_fp32_C_div ( input_fp32_local_addr, input_fp32_local_addr, value.f32, &shape, NULL, NULL );
        tpu_bdc_cast ( output_local_addrs[index], input_fp32_local_addr, &shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
      }
    }
    l2s = true;
    l2s_global_addr = output_global_addr + done * dsize;
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s )
  {
    tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
  }
}

void tpu_kernel_api_cdiv ( const void * args )
{
  sg_api_cdiv_t * api = ( sg_api_cdiv_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 || api->dtype == DT_INT32 );
  scalar_t value;
  if ( api->dtype == DT_FP32 )
  {
    value.f32 = api->value;
  }
  else if (api->dtype == DT_INT32) {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_to_int_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  else
  {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_cop ( api->input_global_addr, api->output_global_addr, value, length, ( data_type_t ) api->dtype, 1 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_cdiv );

void tpu_kernel_api_csub ( const void * args )
{
  sg_api_csub_t * api = ( sg_api_csub_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 || api->dtype == DT_INT32 );
  scalar_t value;
  if ( api->dtype == DT_FP32 )
  {
    value.f32 = api->value;
  }
  else if (api->dtype == DT_INT32) {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_to_int_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  else
  {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_cast ( value_f32, ( data_type_t ) api->dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_cop ( api->input_global_addr, api->output_global_addr, value, length, ( data_type_t ) api->dtype, 0 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_csub );

#ifdef FIRMWARE_BACKEND_2260
extern void nodechip_const_binary_fp_multi_core(
    global_addr_t A_global_addr,
    global_addr_t res_global_addr,
    const int* shape,
    int shape_dim,
    float B_const_val,
    int inversed,
    int binary_type,      // 0: add, 1: sub, 2: mul, 3:div
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

void tpu_kernel_api_const_binary_multi_core(const void* api_buf) {

    sg_api_const_binary_float_t *api = (sg_api_const_binary_float_t*)api_buf;
    tpu_initialize();

    nodechip_const_binary_fp_multi_core(
        api->input_addr,
        api->output_addr,
        api->shape,
        api->dims,
        api->const_value,
        api->is_inversed,
        api->binary_type,
        (data_type_t)(api->dtype),
        0, 0);

    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_const_binary_multi_core);
#endif