#include "sg_api_struct.h"
#include "tpu_kernel.h"


/*
 * output = signbit(input)
 */

void nodechip_signbit (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int length,
data_type_t dtype )
{

  if(length==0) return;
  int npu_num=tpu_npu_num();
  int bank_num=tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu()/bank_num;
  int tensor_num=2+2+0; // 2 inputs, 2 outputs, 1 buffer
  int coeff_bank_num=0; // 0 coeff
  int tensor_size = (bank_num-coeff_bank_num)/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};

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

  scalar_t C = {.u8 = 0};
  scalar_t true_val;
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
    
   // tpu_bdc_and_C(work_local_addr, input_local_addrs[index], neg_C, &shape, NULL, NULL, dtype);
  //  tpu_bdc_equal_C(output_local_addrs[index], work_local_addr, inf_C, C, &shape, NULL, NULL, DT_UINT8, dtype);
    tpu_bdc_less_C(output_local_addrs[index],input_local_addrs[index], C, true_val , &shape,NULL, NULL, DT_UINT8, dtype);

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

int tpu_kernel_api_signbit_multi_core ( const void * args )
{
  sg_api_signbit_t * api = ( sg_api_signbit_t * ) args;
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
#ifdef ENABLE_MULTI_CORE
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  if (core_idx * length_slice < length) {
  nodechip_signbit(
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      cur_length_slice,
      (data_type_t)api->dtype);
  }
  tpu_poll();
  return 0;
#else
  nodechip_signbit ( api->input_global_addr, api->output_global_addr, length, ( data_type_t ) api->dtype );
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_signbit_multi_core );
