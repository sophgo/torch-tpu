#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = sinh(input)
 */
 
inline static void pipeline_move(unsigned long long *array, int num)
{
  for (int i = num - 1; i > 0; i--)
  {
    array[i] = array[i - 1];
  }
}
/**
 * output = log(input)
 */
void nodechip_sinh(global_addr_t out_global_addr,
                  global_addr_t in_global_addr,
                  unsigned long long length,
                  data_type_t dtype
                 )
{
  // 2 bank for work0, 2 bank for work1, 1 bank for coeff and table
  // 2 bank for input, 2 bank for output
  if (length == 0)
  {
    return;
  }
 
  unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
 
  int tensor_num = 2 + 2 + 2 +2 + 1 + 1; // 2 inputs, 2 outputs, 2 work0, 2 wor1, 1 coeff_buffer, 1 table_buffer
  int tensor_bsize_pnpu = tpu_bank_num() / tensor_num * bank_bsize;
  TPUKERNEL_ASSERT(tensor_bsize_pnpu > 0);
 
  local_addr_t in_local_addr[2] = {0, tensor_bsize_pnpu};
  local_addr_t out_local_addr[2] = {2 * tensor_bsize_pnpu, 3 * tensor_bsize_pnpu};
  local_addr_t work0_addr = 4 * tensor_bsize_pnpu;
  local_addr_t work1_addr = 6 * tensor_bsize_pnpu;
  local_addr_t coeff_addr = 7 * tensor_bsize_pnpu;
  local_addr_t table_addr = 8 * tensor_bsize_pnpu;

  int dtype_size = tpu_data_type_size(dtype);
  int npu_num = MIN((unsigned)tpu_npu_num(), length);
  const unsigned int max_m_dim_ = 16384; // hyper parameter,
  // max local memory is tpu_local_mem_size_per_npu() * tpu_npu_num(), in bm1684x, is 16384 * 16 * 64
  // when max_m_dim is 16384, it means both n and m will never exceed 65535
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;
  const unsigned int w_dim = tpu_eu_num(dtype);
  const unsigned int m_dim = MIN( // final used `max_m_dim` should take input into considered.
      MIN(
          length,
          (unsigned int)npu_num * w_dim // w * npu_num (minimum c dim) = m dim
          ),
      max_m_dim_);
 
  const int tensor_size_pnpu = tensor_bsize_pnpu / dtype_size;
  const int n_dim = tensor_size_pnpu * npu_num / m_dim;
 
  tpu_bdc_load_fp32_exp_coeff(coeff_addr);
  tpu_bdc_load_fp32_exp_table(table_addr);
 
  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
 
  // 3. 流水
  // stage_idx 每次循环 + 1
  // draning_idx 每次在数据加载完毕后每次 + 1
 
  // 只要 stage_idx > 0 (已有数据加载进 local memory 就一直 compute)
  // stage_idx
  // 0 1 2 3 4
  // L C S
  //   L C S
  //     L C S
  //       L C S
  // 0 0 0 1 2 3
  // draning_idx           
  // 只要 draning_idx < 1 就一直 load    while (cur_idx[2] < length)
  {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1)
    {
      unsigned long long cur_len = MIN(
          length - cur_idx[0], // remained element size
          n_dim * m_dim        // max matrix element size, n_dim * m_dim < tensor_size_pnpu * npu_num
      );

      // if cur_len is larger than m_dim (for a big matrix), limit `m` to not exceed 65535, and make the matrix more 'square', in this case n > 1
      // else, take cur_len as m, in this case n = 1
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0]; // cur_len / cur_m_dim[0] >= 1
    }
 
    // store output
    if (stage_idx > 1)
    {
      tpu_gdma_matrix_L2S(
          out_global_addr + cur_idx[2] * dtype_size,
          out_local_addr[stage_idx & 0x1],
          /**rows, cols, cols_per_channel, row_stride*/
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2], dtype);
    }
 
    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_matrix_S2L(
          in_local_addr[stage_idx & 0x1],
          in_global_addr + cur_idx[0] * dtype_size,
          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0], dtype);
    }
 
    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1, w_dim};
     
        tpu_bdc_fp32_sinh(out_local_addr[(stage_idx - 1) & 0x1],
                       in_local_addr[(stage_idx - 1) & 0x1],
                       work0_addr,
                       work1_addr,
                       coeff_addr,
                       table_addr,
                       &cur_shape);     
     
    }
 
    tpu_parallel_end();
    // 数组右移
    pipeline_move (cur_idx, 3);
    pipeline_move (cur_m_dim, 3);
    pipeline_move (cur_n_dim, 3);
    if (draning_idx < 1)
    {
      cur_idx[0] += cur_m_dim[0] * cur_n_dim[0];
      if (cur_idx[0] >= length)
      {         
        // draning_idx 每次在数据加载完毕后每次 + 1
        draning_idx++;
      }
    }
    else
    {
      // draning_idx 每次在数据加载完毕后每次 + 1
      draning_idx++;
    }
    // stage_idx 每次循环 + 1
    stage_idx++;    }
}