#include "sg_api_struct.h"
#include "tpu_kernel.h"

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

extern void nodechip_active_local(local_addr_t in_addr, local_addr_t out_addr,
                                  local_addr_t buffer_addr, const int *shape,
                                  data_type_t dtype,
                                  sg_active_type_t active_type,
                                  int if_local_layer, float *coef);

static void nodechip_active(global_addr_t in_global_addr,
                            global_addr_t out_global_addr, const int *shape,
                            int shape_dim, data_type_t dtype,
                            sg_active_type_t active_type, float *coef) {
  unsigned long long length = 1;
  for (int i = 0; i < shape_dim; i++) {
    length *= (unsigned long long)shape[i];
  }

  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_num =
      2 + 2 + 1 + 1; // 2 inputs, 2 outputs, 1 buffer, 1 coeff_buffer
  if (active_type == ACTIVE_ERFC) {
    tensor_num = 2 + 2 + 3 + 2; // 2 inputs, 2 outputs, 3 buffer, 2 coeff_buffer
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

  // for erfc, which need 3 buffer and 2 coeff
  local_addr_t buffer2_addr = 5 * tensor_bsize_pnpu;
  local_addr_t buffer3_addr = 6 * tensor_bsize_pnpu;
  local_addr_t exp_coeff_addr = 7 * tensor_bsize_pnpu;
  local_addr_t erf_coeff_addr = 8 * tensor_bsize_pnpu;
  if (active_type == ACTIVE_ERFC) {
    tpu_bdc_load_fp32_exp_coeff(exp_coeff_addr);
    tpu_bdc_load_fp32_erf_coeff(erf_coeff_addr);
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

      if (active_type == ACTIVE_ERFC) {
        dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                          w_dim}; // matrix layout shape (n, c, h, w)
        tpu_bdc_fp_erfc(out_local_addr[(stage_idx - 1) & 0x1],
                        in_local_addr[(stage_idx - 1) & 0x1], buffer_addr,
                        buffer2_addr, buffer3_addr, exp_coeff_addr,
                        erf_coeff_addr, &cur_shape, dtype);
      } else {

        int cur_shape[4] = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                            w_dim}; // matrix layout shape (n, c, h, w)
        nodechip_active_local(in_local_addr[(stage_idx - 1) & 0x1],
                              out_local_addr[(stage_idx - 1) & 0x1],
                              buffer_addr, cur_shape, dtype, active_type, 0,
                              coef);
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

extern void nodechip_active_multi_core(global_addr_t input_global_addr,
                                       global_addr_t output_global_addr,
                                       const int *shape, int dims,
                                       sg_active_type_t active_type,
                                       float *coeff, data_type_t dtype);

void tpu_kernel_api_active(const void *args) {
  sg_api_active_t *api = (sg_api_active_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
  TPUKERNEL_ASSERT(dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16);
  tpu_initialize();
  nodechip_active(api->input_global_addr, api->output_global_addr, api->shape,
                  api->dim, dtype, api->active_type, NULL);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_active);

void tpu_kernel_api_active_multi_core(const void *args) {
  sg_api_active_t *api = (sg_api_active_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
  TPUKERNEL_ASSERT(dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16);
  tpu_initialize();
  nodechip_active_multi_core(api->input_global_addr, api->output_global_addr,
                             api->shape, api->dim, api->active_type, NULL,
                             dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_active_multi_core);