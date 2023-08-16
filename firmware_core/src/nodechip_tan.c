#include "sg_api_struct.h"
#include "tpu_kernel.h"

inline static void pipeline_move(unsigned long long *array, int num)
{
  for (int i = num - 1; i > 0; i--)
  {
    array[i] = array[i - 1];
  }
}

/*
 * output = input + value * ( other )
 */
void nodechip_tan(global_addr_t out_global_addr,
                  global_addr_t in_global_addr,
                  unsigned long long length,
                  data_type_t dtype)
{
  // 2 bank for work0, 2 bank for work1, 1 bank for coeff and table
  // 2 bank for input, 2 bank for output

  unsigned int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
  local_addr_t in_local_addr[2] = {0, 2 * bank_size};
  local_addr_t out_local_addr[2] = {4 * bank_size, 6 * bank_size};
  local_addr_t buffer_addr = 8 * bank_size;
  local_addr_t coeff_addr = 10 * bank_size;
  tpu_bdc_load_fp32_tan_coeff(coeff_addr);
  int npu_num = tpu_npu_num();
  int eu_num = tpu_eu_num(dtype);
  int tensor_w = MAX(DIV_UP(MIN(length, 2 * bank_size), npu_num), DIV_UP((unsigned)128, eu_num * tpu_data_type_size(dtype)));
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
      tpu_gdma_vector_L2S(
          out_global_addr + cur_idx[2] * tpu_data_type_size(dtype), out_local_addr[stage_idx & 0x1],
          cur_len[2], tensor_w, dtype);
    }

    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_vector_S2L(
          in_local_addr[stage_idx & 0x1], in_global_addr + cur_idx[0] * tpu_data_type_size(dtype),
          cur_len[0], tensor_w, dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      dim4 cur_shape = {1, DIV_UP(cur_len[1], tensor_w), 1, tensor_w};

      tpu_bdc_fp32_tan(out_local_addr[(stage_idx - 1) & 0x1],
                       in_local_addr[(stage_idx - 1) & 0x1],
                       buffer_addr,
                       coeff_addr,
                       &cur_shape);
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

void tpu_kernel_api_tan(const void *args)
{
  sg_api_trifunc_t *api = (sg_api_trifunc_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i)
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_tan(api->output_global_addr, api->input_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_tan);

void tpu_kernel_api_tan_multi_core(const void *args)
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_tan_multi_core);