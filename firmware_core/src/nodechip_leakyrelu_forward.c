#include "sg_api_struct.h"
#include "tpu_kernel.h"


inline static void pipeline_move(unsigned long long *array, int num)
{
  for (int i = num - 1; i > 0; i--)
  {
    array[i] = array[i - 1];
  }
}

void nodechip_leakyrelu (
global_addr_t output_global_addr,
global_addr_t input_global_addr,
scalar_t negative_slope,
unsigned long long length,
data_type_t dtype )
{
  if(length==0) return;
  unsigned int bank_size = tpu_local_mem_size_per_npu()/tpu_bank_num();
  int npu_num = tpu_npu_num();
  int tensor_num = 4; // 2 outputs, 2 input
  int tensor_size = tpu_bank_num()/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t output_local_addr[2] = {0, 2 * bank_size};
  local_addr_t input_local_addr[2] = {4* bank_size, 6 * bank_size};

  int dtype_size = tpu_data_type_size(dtype);
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;

  const unsigned int w_dim = tpu_eu_num(dtype);
  const int tensor_size_pnpu = tensor_size / dtype_size;
  const int n_dim = tensor_size_pnpu * npu_num / max_m_dim;

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
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
          output_global_addr + cur_idx[2] * dtype_size,
          output_local_addr[stage_idx & 0x1],
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],dtype);
    }

    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_matrix_S2L(
          input_local_addr[stage_idx & 0x1],
          input_global_addr + cur_idx[0] * dtype_size,
           cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1, w_dim};
      tpu_bdc_prelu(output_local_addr[(stage_idx - 1) & 0x1],
                    input_local_addr[(stage_idx - 1) & 0x1],
                    negative_slope,
                    &cur_shape, dtype);
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

int tpu_kernel_api_leakyrelu_multi_core(const void *args)
{
  sg_api_leakyrelu_t * api = ( sg_api_leakyrelu_t * ) args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
#ifdef BACKEND_SG2260
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  nodechip_leakyrelu(
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      (scalar_t)api->negative_slope,
      cur_length_slice,
      (data_type_t)api->dtype);

  tpu_poll();
  return 0;
#else
  nodechip_leakyrelu ( api->output_global_addr, api->input_global_addr, (scalar_t)api->negative_slope, length, ( data_type_t ) api->dtype );
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_leakyrelu_multi_core);