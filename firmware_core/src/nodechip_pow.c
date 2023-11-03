#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

inline static void pipeline_move(unsigned long long *array, int num)
{
  for (int i = num - 1; i > 0; i--)
  {
    array[i] = array[i - 1];
  }
}

/*
 * output = math.pow(input, other)
 */
void nodechip_pow(global_addr_t output_global_addr,
                  global_addr_t input_global_addr0,
                  global_addr_t input_global_addr1,
                  unsigned long long length,
                  data_type_t dtype)
{

  if(length==0) return;
  unsigned int bank_size = tpu_local_mem_size_per_npu()/tpu_bank_num();
  int npu_num = tpu_npu_num();
  int tensor_num = 13; // 2 outputs, 2 input1, 2 input2, 2 work1, 2 work2, 1 exp_coeff, 1 log_coeff, 1 exp_table
  int tensor_size = tpu_bank_num()/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t output_local_addr[2] = {0, 1 * bank_size};
  local_addr_t input_local_addr0[2] = {2 * bank_size, 3 * bank_size};
  local_addr_t input_local_addr1[2] = {4 * bank_size, 5 * bank_size};
  local_addr_t work_local_addr0[2] = {6 * tensor_size, 7 * tensor_size};
  local_addr_t work_local_addr1[2] = {8 * tensor_size, 9 * tensor_size};
  local_addr_t exp_coeff_local_addr = 10 * tensor_size;
  local_addr_t log_coeff_local_addr =  11 * tensor_size ;
  local_addr_t exp_table_local_addr = 12 * tensor_size;

  int dtype_size = tpu_data_type_size(dtype);
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;
  const unsigned int w_dim = tpu_eu_num(dtype);
  const int tensor_size_pnpu = tensor_size / dtype_size;
  const int n_dim = tensor_size_pnpu * npu_num / max_m_dim;

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp_log_coeff(log_coeff_local_addr, dtype);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
  while (cur_idx[2] < length)
  {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1)
    {
       unsigned long long cur_len = MIN(
          length - cur_idx[0],
          n_dim * max_m_dim
      );
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0];
    }

    // store output
    if (stage_idx > 1)
    {
      tpu_gdma_matrix_L2S(
          output_global_addr + cur_idx[2] * tpu_data_type_size(dtype),
          output_local_addr[stage_idx & 0x1],
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2], dtype);
    }

    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_matrix_S2L(
          input_local_addr0[stage_idx & 0x1],
          input_global_addr0 + cur_idx[0] * tpu_data_type_size(dtype),
           cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0], dtype);
      tpu_gdma_matrix_S2L(
          input_local_addr1[stage_idx & 0x1],
          input_global_addr1 + cur_idx[0] * tpu_data_type_size(dtype),
           cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0], dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1, w_dim};
      tpu_bdc_fp32_pow(output_local_addr[(stage_idx - 1) & 0x1],
                    input_local_addr0[(stage_idx - 1) & 0x1],
                    input_local_addr1[(stage_idx - 1) & 0x1],
                    work_local_addr0[(stage_idx - 1) & 0x1],
                    work_local_addr1[(stage_idx - 1) & 0x1],
                    exp_coeff_local_addr, log_coeff_local_addr, exp_table_local_addr,
                    &cur_shape);
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_m_dim, 3);
    pipeline_move(cur_n_dim, 3);
    if (draning_idx < 1)
    {
      cur_idx[0] += cur_m_dim[0] * cur_n_dim[0];
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

void tpu_kernel_api_pow(const void *args)
{
  sg_api_pow_t *api = (sg_api_pow_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i)
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_pow(api->output_global_addr, api->input_global_addr, api->other_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_pow_multi_core(const void *args)
{
   sg_api_pow_t * api = ( sg_api_pow_t * ) args;
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
  nodechip_pow(
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->other_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      cur_length_slice,
      (data_type_t)api->dtype);

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_multi_core);
#endif