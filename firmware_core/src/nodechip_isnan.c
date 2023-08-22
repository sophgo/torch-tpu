#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = isnan(input)
 */

void nodechip_isnan (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int length,
data_type_t dtype )
{

  if(length==0) return;
  int npu_num=tpu_npu_num();
  int bank_num=tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu()/bank_num;
  int tensor_num=2+2+3; // 2 inputs, 2 outputs, 3 buffer
  int coeff_bank_num=0; // 0 coeff
  int tensor_size = (bank_num-coeff_bank_num)/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t work_local_addr[3] = {4 * tensor_size, 5 * tensor_size, 6 * tensor_size};

  int dtype_size = tpu_data_type_size(dtype);
  int tensor_w = DIV_UP(MIN(length, tensor_size*npu_num/dtype_size), npu_num);

  int todo = length;
  int done = 0;
  dim4 shape = { .n = 1, .h = 1 };
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;

  scalar_t inf_C = {.u32 = (dtype == DT_FP32 ? 0x7f800000 : (dtype == DT_FP16 ? 0x7c00 : 0x7f80))};
  scalar_t neg_C= {.u32 = (dtype == DT_FP32 ? 0x7fffffff : (dtype == DT_FP16 ? 0x7fff : 0x7fff))};
  scalar_t  C= {.u8 = 1};

  while ( todo != 0 )
  {
    if ( todo > NPU_NUM )
    {
      shape.c = NPU_NUM;
      shape.w = MIN ( todo / NPU_NUM, tensor_w );
    }
    else
    {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dtype_size, &shape, NULL, NULL, dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, DT_UINT8 );
    }
    
    tpu_bdc_and_C(work_local_addr[2], input_local_addrs[index], inf_C, &shape, NULL, NULL, dtype);
    tpu_bdc_equal_C(work_local_addr[0], work_local_addr[2], inf_C, C, &shape, NULL, NULL, DT_UINT8, dtype);

    tpu_bdc_and_C(work_local_addr[2], input_local_addrs[index], neg_C, &shape, NULL, NULL, dtype);
    tpu_bdc_not_equal_C(work_local_addr[1], work_local_addr[2], inf_C, C, &shape, NULL, NULL, DT_UINT8, dtype);
    
    tpu_bdc_and(work_local_addr[2],work_local_addr[0],work_local_addr[1],&shape,NULL,NULL,NULL,DT_UINT8);
    tpu_bdc_equal_C(output_local_addrs[index], work_local_addr[2], C, C, &shape, NULL, NULL, DT_UINT8, DT_UINT8);

    l2s = true;
    l2s_global_addr = output_global_addr + done;
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
    tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, DT_UINT8 );
  }
}

void tpu_kernel_api_isnan ( const void * args )
{
  sg_api_isnan_t * api = ( sg_api_isnan_t * ) args;
  
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_isnan ( api->input_global_addr, api->output_global_addr, length, ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_isnan );

void tpu_kernel_api_isnan_multi_core ( const void * args )
{
  sg_api_isnan_t * api = ( sg_api_isnan_t * ) args;
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
  nodechip_isnan(
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      cur_length_slice,
      (data_type_t)api->dtype);

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_isnan_multi_core );