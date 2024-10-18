#include "sg_api_struct.h"
#include "tpu_kernel.h"


inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

/*
 * output = input + value * ( tensor1 / tensor2 )
 */

void nodechip_addcdiv(global_addr_t input_global_addr,
                      global_addr_t tensor1_global_addr,
                      global_addr_t tensor2_global_addr,
                      global_addr_t output_global_addr, scalar_t value,
                      unsigned long long length, data_type_t dtype) {
  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(DT_FP32);
  int tensor_num =
      dtype != DT_FP32 ? 12 : 8; // if dtype != DT_FP32, need additional addr
  int tensor_bsize_pnpu = tpu_bank_num() / tensor_num * bank_bsize;
  TPUKERNEL_ASSERT(tensor_bsize_pnpu > 0);

  // max local memory is tpu_local_mem_size_per_npu() * tpu_npu_num()
  // for bm1684x, is (16384   *   16)    *   64
  //                    ↑          ↑          ↑
  //                bank_size * bank_num * npu_num
  // (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1 = 32768,
  // when `m_dim` is set to this value, it ensures both n and m dim to not
  // exceed **shape limit**
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;

  // w should larger equal than tpu_eu_num(dtype) to make full use of
  // eu(execution unit)
  const unsigned int w_dim = tpu_eu_num(dtype);
  const int n_dim = tensor_bsize_pnpu * tpu_npu_num() / dtype_size / max_m_dim;

  local_addr_t in_local_addr[2] = {0, 1 * tensor_bsize_pnpu};
  local_addr_t tensor1_local_addr[2] = {2 * tensor_bsize_pnpu,
                                        3 * tensor_bsize_pnpu};
  local_addr_t tensor2_local_addr[2] = {4 * tensor_bsize_pnpu,
                                        5 * tensor_bsize_pnpu};
  local_addr_t out_local_addr[2] = {6 * tensor_bsize_pnpu,
                                    7 * tensor_bsize_pnpu};

  local_addr_t tensor1_fp32_local_addr[2], tensor2_fp32_local_addr[2];
  if (dtype != DT_FP32) {
    tensor1_fp32_local_addr[0] = 8 * tensor_bsize_pnpu;
    tensor1_fp32_local_addr[1] = 9 * tensor_bsize_pnpu;
    tensor2_fp32_local_addr[0] = 10 * tensor_bsize_pnpu;
    tensor2_fp32_local_addr[1] = 11 * tensor_bsize_pnpu;
  } else {
    tensor1_fp32_local_addr[0] = tensor1_local_addr[0];
    tensor1_fp32_local_addr[1] = tensor1_local_addr[1];
    tensor2_fp32_local_addr[0] = tensor2_local_addr[0];
    tensor2_fp32_local_addr[1] = tensor2_local_addr[1];
  }

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length) {
    tpu_parallel_start();

    if (draning_idx < 1) {
      unsigned long long cur_len =
          MIN(length - cur_idx[0], // remained element size
              n_dim * max_m_dim    // max matrix element size, n_dim * m_dim <
                                   // tensor_size_pnpu * npu_num
          );

      // if cur_len is larger than m_dim (for a big matrix), limit `m` to not
      // exceed max_m_dim, in this case n > 1 else, take cur_len as m, in this
      // case n = 1 NOTE: n_dim * max_m_dim <= tensor_size_pnpu * npu_num, it's
      // always a legal size.
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0]; // cur_len / cur_m_dim[0] >= 1
    }

    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(output_global_addr + cur_idx[2] * dtype_size,
                          out_local_addr[stage_idx & 0x1],
                          /**rows, cols, cols_per_channel, row_stride*/
                          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],
                          DT_FP32);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(in_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
      tpu_gdma_matrix_S2L(tensor1_local_addr[stage_idx & 0x1],
                          tensor1_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
      tpu_gdma_matrix_S2L(tensor2_local_addr[stage_idx & 0x1],
                          tensor2_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)

      if (dtype != DT_FP32) {
        tpu_bdc_cast(tensor1_fp32_local_addr[(stage_idx - 1) & 0x1],
                     tensor1_local_addr[(stage_idx - 1) & 0x1], &cur_shape,
                     NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        tpu_bdc_cast(tensor2_fp32_local_addr[(stage_idx - 1) & 0x1],
                     tensor2_local_addr[(stage_idx - 1) & 0x1], &cur_shape,
                     NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        tpu_bdc_fp32_div(tensor1_fp32_local_addr[(stage_idx - 1) & 0x1],
                         tensor1_fp32_local_addr[(stage_idx - 1) & 0x1],
                         tensor2_fp32_local_addr[(stage_idx - 1) & 0x1],
                         &cur_shape, NULL, NULL, NULL);
        tpu_bdc_cast(out_local_addr[(stage_idx - 1) & 0x1],
                     tensor1_fp32_local_addr[(stage_idx - 1) & 0x1], &cur_shape,
                     NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN);
      } else {
        tpu_bdc_fp32_div(out_local_addr[(stage_idx - 1) & 0x1],
                         tensor1_fp32_local_addr[(stage_idx - 1) & 0x1],
                         tensor2_fp32_local_addr[(stage_idx - 1) & 0x1],
                         &cur_shape, NULL, NULL, NULL);
      }
      tpu_bdc_fp_mul_C(out_local_addr[(stage_idx - 1) & 0x1],
                       out_local_addr[(stage_idx - 1) & 0x1], value, &cur_shape,
                       NULL, NULL, dtype);
      tpu_bdc_fp_add(out_local_addr[(stage_idx - 1) & 0x1],
                     out_local_addr[(stage_idx - 1) & 0x1],
                     in_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, NULL, dtype);
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_m_dim, 3);
    pipeline_move(cur_n_dim, 3);
    if (draning_idx < 1) {
      cur_idx[0] += cur_m_dim[0] * cur_n_dim[0];
      if (cur_idx[0] >= length) {
        draning_idx++;
      }
    } else {
      draning_idx++;
    }
    stage_idx++;
  }
}

int tpu_kernel_api_addcdiv(const void *args) {
  sg_api_addcdiv_t *api = (sg_api_addcdiv_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
  TPUKERNEL_ASSERT(dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16);
  scalar_t value;
  if (dtype == DT_FP32) {
    value.f32 = api->value;
  } else {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, dtype, DT_FP32, RM_HALF_TO_EVEN);
  }
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_addcdiv(api->input_global_addr, api->tensor1_global_addr,
                   api->tensor2_global_addr, api->output_global_addr, value,
                   length, dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_addcdiv);
