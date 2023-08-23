#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = arctan(input)
 * arctan(x)=arcsin(x/sqrt(x^2+1))
 */

void nodechip_atan (
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
  int coeff_bank_num= 4 ; // 4 coeff
  int tensor_size = (bank_num-coeff_bank_num)/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t work_local_addrs[3] = {4 * tensor_size, 5 * tensor_size, 6 * tensor_size};
  local_addr_t exp_coeff_local_addr = 7 * tensor_size;
  local_addr_t log_coeff_local_addr = 7 * tensor_size + 1 * bank_size; 
  local_addr_t exp_table_local_addr = 7 * tensor_size + 2 * bank_size; 
  local_addr_t arcsin_coeff_local_addr = 7 * tensor_size + 3 * bank_size; 

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
  float POW_C=2;
  scalar_t ADD_C = {.f32 = 1.f};
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
  tpu_bdc_load_fp32_arcsin_coeff(arcsin_coeff_local_addr);

  while ( todo != 0 )
  {
    if ( todo > NPU_NUM )
    {
      shape.c = NPU_NUM;
      shape.w = MIN ( todo / NPU_NUM,tensor_w );
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
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    
    tpu_bdc_abs(work_local_addrs[2],input_local_addrs[index],&shape,NULL,NULL,dtype);
    tpu_bdc_fp32_pow_C(output_local_addrs[index],work_local_addrs[2],work_local_addrs[0],work_local_addrs[1],
                              exp_coeff_local_addr,log_coeff_local_addr,exp_table_local_addr,POW_C,&shape);
    tpu_bdc_fp_add_C(work_local_addrs[2],output_local_addrs[index], ADD_C, &shape, NULL, NULL, dtype); 
    tpu_bdc_fp32_rsqrt(output_local_addrs[index],work_local_addrs[2],&shape) ;
    tpu_bdc_fp_mul(work_local_addrs[2],output_local_addrs[index],input_local_addrs[index],&shape,NULL,NULL,NULL,dtype);
    tpu_bdc_fp32_arcsin(output_local_addrs[index],work_local_addrs[2], work_local_addrs[0],arcsin_coeff_local_addr,&shape);

    l2s = true;
    l2s_global_addr = output_global_addr + done * dtype_size;
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

void tpu_kernel_api_atan ( const void * args )
{
  sg_api_atan_t * api = ( sg_api_atan_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32);
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_atan ( api->input_global_addr, api->output_global_addr, length, ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_atan );

void tpu_kernel_api_atan_multi_core ( const void * args )
{
  sg_api_atan_t * api = ( sg_api_atan_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32);

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
  nodechip_atan(
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      cur_length_slice,
      (data_type_t)api->dtype);

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_atan_multi_core );