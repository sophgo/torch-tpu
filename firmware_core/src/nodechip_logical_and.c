#include "sg_api_struct.h"
#include "tpu_kernel.h"


/*
 * output = input && other
 */

inline static void pipeline_move(unsigned long long *array, int num)
{
  for (int i = num - 1; i > 0; i--)
  {
    array[i] = array[i - 1];
  }
}

void nodechip_logical_and(
    global_addr_t output_global_addr,
    global_addr_t input_global_addr0,
    global_addr_t input_global_addr1,
    unsigned long long length,
    data_type_t dtype)
{
  if(length==0) return;
  unsigned int bank_size = tpu_local_mem_size_per_npu()/tpu_bank_num();
  int npu_num = tpu_npu_num();
  int tensor_num = 6; // 2 outputs, 2 input1, 2 input2
  int tensor_size = tpu_bank_num()/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t output_local_addr[2] = {0, 2 * bank_size};
  local_addr_t input_local_addr0[2] = {4 * bank_size, 6 * bank_size};
  local_addr_t input_local_addr1[2] = {8 * bank_size, 10 * bank_size};

  int dtype_size = tpu_data_type_size(dtype);
  int tensor_w = DIV_UP(MIN(length, (unsigned long long)tensor_size*npu_num/dtype_size), npu_num);
  unsigned long long slice = MIN(length, (unsigned long long)npu_num * tensor_w);

  unsigned long long cur_idx[3] = {0}, cur_len[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length)
  {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1)
    {
      cur_len[0] = MIN(length - cur_idx[0], slice);
    }

    // store output
    if (stage_idx > 1)
    {
      tpu_gdma_matrix_L2S(
          output_global_addr + cur_idx[2] * tpu_data_type_size(dtype),
          output_local_addr[stage_idx & 0x1],
          1, cur_len[2], tensor_w,
          1, dtype);
    }

    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_matrix_S2L(
          input_local_addr0[stage_idx & 0x1],
          input_global_addr0 + cur_idx[0] * tpu_data_type_size(dtype),
          1, cur_len[0], tensor_w,
          1, dtype);
      tpu_gdma_matrix_S2L(
          input_local_addr1[stage_idx & 0x1],
          input_global_addr1 + cur_idx[0] * tpu_data_type_size(dtype),
          1, cur_len[0], tensor_w,
          1, dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      dim4 cur_shape = {1, DIV_UP(cur_len[1], tensor_w), 1, tensor_w};

      tpu_bdc_and(output_local_addr[(stage_idx - 1) & 0x1],
                    input_local_addr0[(stage_idx - 1) & 0x1],
                    input_local_addr1[(stage_idx - 1) & 0x1],
                    &cur_shape, NULL, NULL ,NULL ,dtype);
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_len, 3);
    if (draning_idx < 1)
    {
      cur_idx[0] += cur_len[0];
      if (cur_idx[0] >= length)
      {
        draning_idx++;
      }
    }
    else
    {
      draning_idx++;
    }
    stage_idx++;
  }

}

int tpu_kernel_api_logical_and_multi_core(const void *args)
{
  sg_api_logical_and_t * api = ( sg_api_logical_and_t * ) args;
#ifdef ENABLE_MULTI_CORE
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
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
  nodechip_logical_and(
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->other_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      cur_length_slice,
      (data_type_t)api->dtype);

  tpu_poll();
  return 0;
#else
  int length = 1;
  for (int i = 0; i < api->dim; ++i)
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_logical_and(api->output_global_addr, api->input_global_addr, api->other_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_logical_and_multi_core);