#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = input + value * ( other )
 */

void nodechip_scale_add (
global_addr_t input_global_addr,
global_addr_t other_global_addr,
global_addr_t output_global_addr,
scalar_t value,
int length,
data_type_t dtype )
{
  const int dsize = tpu_data_type_size ( dtype );
  int wmax = DIV_UP ( length, NPU_NUM );
  local_addr_t input_local_addrs[2], other_local_addrs[2], output_local_addrs[2];
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = tpu_aligned_feature_size ( 1, wmax, dtype );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    other_local_addrs[0] = next; next += size;
    other_local_addrs[1] = next; next += size;
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
    tpu_gdma_cpy_S2L ( other_local_addrs[index], other_global_addr + done * dsize, &shape, NULL, NULL, dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    tpu_bdc_fp_mul_C ( other_local_addrs[index], other_local_addrs[index], value, &shape, NULL, NULL, dtype );
    tpu_bdc_fp_add ( output_local_addrs[index], input_local_addrs[index], other_local_addrs[index], &shape, NULL, NULL, NULL, dtype );
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

void tpu_kernel_api_add ( const void * args )
{
  sg_api_add_t * api = ( sg_api_add_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  scalar_t value;
  if ( api->dtype == DT_FP32 )
  {
    value.f32 = api->value;
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
  nodechip_scale_add ( api->input_global_addr, api->other_global_addr, api->output_global_addr, value, length, ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_add );

void tpu_kernel_api_add_multi_core ( const void * args )
{
  sg_api_add_t * api = ( sg_api_add_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  scalar_t value;
  if ( api->dtype == DT_FP32 )
  {
    value.f32 = api->value;
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

  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();

  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);

  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  nodechip_scale_add(
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->other_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      value,
      cur_length_slice,
      (data_type_t)api->dtype);

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_add_multi_core );