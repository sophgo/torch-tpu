#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = input + value * ( tensor1 / tensor2 )
 */

void nodechip_addcdiv (
global_addr_t input_global_addr,
global_addr_t tensor1_global_addr,
global_addr_t tensor2_global_addr,
global_addr_t output_global_addr,
scalar_t value,
int length,
data_type_t dtype )
{
  const int dsize = tpu_data_type_size ( dtype );
  int wmax = DIV_UP ( length, NPU_NUM );
  local_addr_t input_local_addrs[2], tensor1_local_addrs[2], tensor2_local_addrs[2], output_local_addrs[2];
  local_addr_t tensor1_fp32_local_addrs[2], tensor2_fp32_local_addrs[2];
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = tpu_aligned_feature_size ( 1, wmax, dtype );
    int size_fp32 = tpu_aligned_feature_size ( 1, wmax, DT_FP32 );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    tensor1_local_addrs[0] = next; next += size;
    tensor1_local_addrs[1] = next; next += size;
    tensor2_local_addrs[0] = next; next += size;
    tensor2_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += size;
    output_local_addrs[1] = next; next += size;
    if ( dtype != DT_FP32 )
    {
      tensor1_fp32_local_addrs[0] = next; next += size_fp32;
      tensor1_fp32_local_addrs[1] = tensor1_fp32_local_addrs[0];
      tensor2_fp32_local_addrs[0] = next; next += size_fp32;
      tensor2_fp32_local_addrs[1] = tensor2_fp32_local_addrs[0];
    }
    else
    {
      tensor1_fp32_local_addrs[0] = tensor1_local_addrs[0];
      tensor1_fp32_local_addrs[1] = tensor1_local_addrs[1];
      tensor2_fp32_local_addrs[0] = tensor2_local_addrs[0];
      tensor2_fp32_local_addrs[1] = tensor2_local_addrs[1];
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
    tpu_gdma_cpy_S2L ( tensor1_local_addrs[index], tensor1_global_addr + done * dsize, &shape, NULL, NULL, dtype );
    tpu_gdma_cpy_S2L ( tensor2_local_addrs[index], tensor2_global_addr + done * dsize, &shape, NULL, NULL, dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast ( tensor1_fp32_local_addrs[index], tensor1_local_addrs[index], &shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
      tpu_bdc_cast ( tensor2_fp32_local_addrs[index], tensor2_local_addrs[index], &shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
    }
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_fp32_div ( tensor1_fp32_local_addrs[index], tensor1_fp32_local_addrs[index], tensor2_fp32_local_addrs[index], &shape, NULL, NULL, NULL );
      tpu_bdc_cast ( output_local_addrs[index], tensor1_fp32_local_addrs[index], &shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
    }
    else
    {
      tpu_bdc_fp32_div ( output_local_addrs[index], tensor1_fp32_local_addrs[index], tensor2_fp32_local_addrs[index], &shape, NULL, NULL, NULL );
    }
    tpu_bdc_fp_mul_C ( output_local_addrs[index], output_local_addrs[index], value, &shape, NULL, NULL, dtype );
    tpu_bdc_fp_add ( output_local_addrs[index], output_local_addrs[index], input_local_addrs[index], &shape, NULL, NULL, NULL, dtype );
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

void tpu_kernel_api_addcdiv ( const void * args )
{
  sg_api_addcdiv_t * api = ( sg_api_addcdiv_t * ) args;
  data_type_t dtype = ( data_type_t ) api->dtype;
  TPUKERNEL_ASSERT ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 );
  scalar_t value;
  if ( dtype == DT_FP32 )
  {
    value.f32 = api->value;
  }
  else
  {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_cast ( value_f32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_addcdiv ( api->input_global_addr, api->tensor1_global_addr, api->tensor2_global_addr, api->output_global_addr, value, length, dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_addcdiv );
