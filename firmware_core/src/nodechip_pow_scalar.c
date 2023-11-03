#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

#define BOFFSET(index) buffer_addr + index *tensor_bsize_pnpu

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

// 2+2 + 1 = 5
void tpu_bdc_fp_pow_234(local_addr_t dst_addr, local_addr_t src_addr,
                        local_addr_t buffer_addr, const dim4 *shape, float C,
                        data_type_t dtype) {
  if (C == 2) {
    buffer_addr = dst_addr;
  }
  tpu_bdc_fp_mul(buffer_addr, src_addr, src_addr, shape, NULL, NULL, NULL,
                 dtype);

  if (C == 4.0) {
    tpu_bdc_fp_mul(dst_addr, buffer_addr, buffer_addr, shape, NULL, NULL, NULL,
                   dtype);
  } else if (C == 3.0) {
    tpu_bdc_fp_mul(dst_addr, buffer_addr, src_addr, shape, NULL, NULL, NULL,
                   dtype);
  }
}
// (2+2) + 2 + 3 = 9
// (2+2) + 2 + 2 + 2 + 2 + 3 = 15
void tpu_bdc_fp_pow_C(local_addr_t dst_addr, local_addr_t src_addr,
                      local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                      local_addr_t work0_addr, local_addr_t work1_addr,
                      local_addr_t exp_coeff_addr, local_addr_t log_coeff_addr,
                      local_addr_t exp_table_addr, float C, const dim4 *shape,
                      data_type_t dtype) {
  if (dtype == DT_FP32) {
    tpu_bdc_fp32_pow_C(dst_addr, src_addr, work0_addr, work1_addr,
                       exp_coeff_addr, log_coeff_addr, exp_table_addr, C,
                       shape);
  } else {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_AWAY_FROM_ZERO);
    tpu_bdc_fp32_pow_C(dst_fp32_addr, src_fp32_addr, work0_addr, work1_addr,
                       exp_coeff_addr, log_coeff_addr, exp_table_addr, C,
                       shape);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                 RM_HALF_AWAY_FROM_ZERO);
  }
}

static inline void nodechip_pow_c_parallel(global_addr_t in_global_addr,
                                           global_addr_t out_global_addr,
                                           unsigned long long length, float C,
                                           data_type_t dtype) {

  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_num = 2 + 2; //  2 inputs, 2 outputs
  bool pow_234 = false;
  if (C == 2 || C == 3 || C == 4) {
    tensor_num += 1;
    pow_234 = true;
  } else if (dtype == DT_FP32) {
    tensor_num += (2 + 3);
  } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
    // dst_fp32, src_fp32, f32_work0, f32_work1, 3 coeff
    tensor_num += (2 + 2 + 2 + 2 + 3);
  } else {
    TPUKERNEL_ASSERT(false);
  }

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
  local_addr_t out_local_addr[2] = {2 * tensor_bsize_pnpu,
                                    3 * tensor_bsize_pnpu};
  local_addr_t buffer_addr = 4 * tensor_bsize_pnpu;

  if (!pow_234) {
    tpu_bdc_load_fp32_exp_coeff(BOFFSET(0));
    tpu_bdc_load_fp32_log_coeff(BOFFSET(1));
    tpu_bdc_load_fp32_exp_table(BOFFSET(2));
  }

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length) {
    tpu_parallel_start();

    // update load info
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
      tpu_gdma_matrix_L2S(out_global_addr + cur_idx[2] * dtype_size,
                          out_local_addr[stage_idx & 0x1],
                          /**rows, cols, cols_per_channel, row_stride*/
                          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],
                          dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(in_local_addr[stage_idx & 0x1],
                          in_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {

      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)

      if (pow_234) {
        tpu_bdc_fp_pow_234(out_local_addr[(stage_idx - 1) & 0x1],
                           in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(0),
                           &cur_shape, C, dtype);
      } else if (dtype == DT_FP32) {
        tpu_bdc_fp_pow_C(out_local_addr[(stage_idx - 1) & 0x1],
                         in_local_addr[(stage_idx - 1) & 0x1], 0, 0, BOFFSET(3),
                         BOFFSET(4), BOFFSET(0), BOFFSET(1), BOFFSET(2), C,
                         &cur_shape, dtype);
      } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
        tpu_bdc_fp_pow_C(out_local_addr[(stage_idx - 1) & 0x1],
                         in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(3),
                         BOFFSET(5), BOFFSET(7), BOFFSET(9), BOFFSET(0),
                         BOFFSET(1), BOFFSET(2), C, &cur_shape, dtype);
      }
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

void tpu_kernel_api_pow_c(const void *args) {
  sg_api_pow_tensor_scalar_t *api = (sg_api_pow_tensor_scalar_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_pow_c_parallel(api->self_global_addr, api->out_global_addr, length,
                          api->value, api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_c);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_pow_c_multi_core(const void *args) {
  sg_api_pow_tensor_scalar_t *api = (sg_api_pow_tensor_scalar_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  unsigned int slice_num = tpu_core_num();
  unsigned int slice_idx = tpu_core_index();
  TPUKERNEL_ASSERT(slice_num > 0);
  TPUKERNEL_ASSERT(0 <= slice_idx && slice_idx < slice_num);
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; i++) {
    length *= (unsigned long long)api->shape[i];
  }

  unsigned long long slice = DIV_UP(length, slice_num);
  if (length < slice_num) {
    slice = 1;
  }
  unsigned int offset = slice_idx * slice;
  unsigned long long real_slice = MIN(slice, length - offset);
  if (real_slice <= 0)
    return;
  const int dsize = tpu_data_type_size(dtype);
  tpu_initialize();
  nodechip_pow_c_parallel(api->self_global_addr + offset * dsize,
                          api->out_global_addr + offset * dsize, real_slice,
                          api->value, api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_c_multi_core);
#endif