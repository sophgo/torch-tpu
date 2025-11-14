#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "kernel_utils_func.h"


#define BOFFSET(index) buffer_addr + index *tensor_bsize_pnpu

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

void tpu_bdc_fp_trunc(local_addr_t dst_addr, local_addr_t src_addr,
                      const dim4 *shape, data_type_t dtype) {
  tpu_bdc_fp_round(dst_addr, src_addr, shape, NULL, NULL, dtype,
                   RM_TOWARDS_ZERO);
}

void tpu_bdc_fp_exp2(local_addr_t dst_addr, local_addr_t src_addr,
                     local_addr_t work0_addr, local_addr_t work1_addr,
                     local_addr_t coeff_addr, local_addr_t table_addr,
                     const dim4 *shape, data_type_t dtype) {
  tpu_bdc_fp32_C_pow(dst_addr, src_addr, work0_addr, work1_addr, coeff_addr,
                     table_addr, /*float C*/ 2., shape);
}

void tpu_bdc_fp_reciprocal_v2(local_addr_t dst_addr, local_addr_t src_addr,
                              local_addr_t dst_fp32_addr,
                              local_addr_t src_fp32_addr, const dim4 *shape,
                              data_type_t dtype) {
  if (dtype != DT_FP32) {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_TO_EVEN);
    tpu_bdc_fp32_tunable_reciprocal(dst_fp32_addr, src_fp32_addr, shape, NULL,
                                    NULL, 3);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                 RM_HALF_TO_EVEN);
    return;
  }
  if (dtype == DT_FP32) {
    tpu_bdc_fp32_tunable_reciprocal(dst_addr, src_addr, shape, NULL, NULL, 3);
    return;
  }
}

void _tpu_bdc_fp_sigmoid(local_addr_t dst_addr, local_addr_t src_addr,
                        local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                        local_addr_t work0_addr, local_addr_t work1_addr,
                        local_addr_t coeff_addr, local_addr_t table_addr,
                        const dim4 *shape, data_type_t dtype) {

  if (dtype != DT_FP32) {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_TO_EVEN);
    tpu_bdc_fp32_sigmoid(dst_fp32_addr, src_fp32_addr, work0_addr, work1_addr,
                         coeff_addr, table_addr, shape);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                 RM_HALF_TO_EVEN);
    return;
  }
  if (dtype == DT_FP32) {
    tpu_bdc_fp32_sigmoid(dst_addr, src_addr, work0_addr, work1_addr, coeff_addr,
                         table_addr, shape);
    return;
  }
  TPUKERNEL_ASSERT(false);
}

/**
 * tpu_bdc_fp_sqrt only support DT_FP32
 */
#ifdef ENABLE_MULTI_CORE
void tpu_bdc_fp_sqrt_v2(local_addr_t dst_addr, local_addr_t src_addr,
                        local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                        const dim4 *shape, data_type_t dtype) {
  tpu_bdc_fp_sqrt(dst_addr, src_addr, shape, DT_FP32);
}
#else
void tpu_bdc_fp_sqrt_v2(local_addr_t dst_addr, local_addr_t src_addr,
                        local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                        const dim4 *shape, data_type_t dtype) {
  if (dtype != DT_FP32) {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_TO_EVEN);
    tpu_bdc_fp_sqrt(dst_fp32_addr, src_fp32_addr, shape, DT_FP32);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                 RM_HALF_TO_EVEN);
    return;
  }
  if (dtype == DT_FP32) {
    tpu_bdc_fp_sqrt(dst_addr, src_addr, shape, DT_FP32);
    return;
  }
}
#endif

/**
 * tpu_bdc_fp32_silu only support DT_FP32
 */
static
void _tpu_bdc_fp_silu(local_addr_t dst_addr, local_addr_t src_addr,
                     local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                     local_addr_t work0_addr, local_addr_t work1_addr,
                     local_addr_t coeff_addr, local_addr_t table_addr,
                     const dim4 *shape, data_type_t dtype) {
  if (dtype != DT_FP32) {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_TO_EVEN);
    tpu_bdc_fp32_silu(dst_fp32_addr, src_fp32_addr, work0_addr, work1_addr,
                      coeff_addr, table_addr, shape);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                 RM_HALF_TO_EVEN);
    return;
  }
  if (dtype == DT_FP32) {
    tpu_bdc_fp32_silu(dst_addr, src_addr, work0_addr, work1_addr, coeff_addr,
                      table_addr, shape);
    return;
  }
}

extern void nodechip_active_local(local_addr_t in_addr, local_addr_t out_addr,
                                  local_addr_t buffer_addr, const int *shape,
                                  data_type_t dtype,
                                  sg_active_type_t active_type,
                                  int if_local_layer, float *coef);

void nodechip_active_v2(global_addr_t in_global_addr,
                        global_addr_t out_global_addr,
                        unsigned long long length, data_type_t dtype,
                        sg_active_type_t active_type, float *coef) {

  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  data_type_t out_dtype = (active_type == ACTIVE_ISINF ||
                     active_type == ACTIVE_ISNAN) ? DT_UINT8 : dtype;
  int output_size = tpu_data_type_size(out_dtype);
  int tensor_num = 2 + 2; //  2 inputs, 2 outputs

  if (active_type == ACTIVE_ERFC) {
    // erfc need 3 buffer and 2 coeff
    tensor_num += 3 + 2;
  } else if (active_type == ACTIVE_EXPM1) {
    // extra 2 input fp 32 buf, 2 output fp32 buf, 1 coeff
    tensor_num += 2 + 2 + 1;
  } else if (active_type == ACTIVE_SIGMOID || active_type == ACTIVE_SILU ||
             active_type == ACTIVE_RECIPROCAL) {
    if (dtype == DT_FP16 || dtype == DT_BFP16) {
      // extra 2 input fp32 buf, 2 output fp32 buf
      // 4 fp32 work addr, 2 coeff_buffer
      tensor_num += 2 + 2 + 4 + 2;
    } else {
      tensor_num += 2 + 2; // extra 2 buffer, 2 coeff_buffer
    }
  } else if (active_type == ACTIVE_ISNAN) {
    // extra 3 buffers
    tensor_num += 3;
  } else if (active_type == ACTIVE_ISINF) {
    // extra 1 buffer
    tensor_num += 1;
  } else {
    // extra 1 buffer, 1 coeff_buffer (common active imp)
    tensor_num += 1 + 1;
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

  if (active_type == ACTIVE_ERFC) {
    tpu_bdc_load_fp_exp_coeff(BOFFSET(0), dtype);
    tpu_bdc_load_fp32_erf_coeff(BOFFSET(1));
  } else if (active_type == ACTIVE_EXPM1) {
    tpu_bdc_load_fp_exp_coeff(BOFFSET(0), dtype);
  } else if (active_type == ACTIVE_SIGMOID || active_type == ACTIVE_SILU) {
    tpu_bdc_load_fp32_exp_coeff(BOFFSET(0));
    tpu_bdc_load_fp32_exp_table(BOFFSET(1));
  } else if (active_type == ACTIVE_EXP2) {
    tpu_bdc_load_fp32_exp_coeff(BOFFSET(0));
    tpu_bdc_load_fp32_exp_table(BOFFSET(1));
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
      tpu_gdma_matrix_L2S(out_global_addr + cur_idx[2] * output_size,
                          out_local_addr[stage_idx & 0x1],
                          /**rows, cols, cols_per_channel, row_stride*/
                          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],
                          out_dtype);
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
      if (active_type == ACTIVE_ERFC) {
        tpu_bdc_fp_erfc(out_local_addr[(stage_idx - 1) & 0x1],
                        in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(2),
                        BOFFSET(3), BOFFSET(4), /* exp coeff */ BOFFSET(0),
                        /* erf coeff */ BOFFSET(1), &cur_shape, dtype);
      } else if (active_type == ACTIVE_SQRT) {
        tpu_bdc_fp_sqrt_v2(out_local_addr[(stage_idx - 1) & 0x1],
                           in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(0),
                           BOFFSET(2), &cur_shape, dtype);
      } else if (active_type == ACTIVE_EXPM1) {
        tpu_bdc_fp_expm1(out_local_addr[(stage_idx - 1) & 0x1],
                         in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(1),
                         BOFFSET(3), BOFFSET(0), &cur_shape, dtype);
      } else if (active_type == ACTIVE_RECIPROCAL) {
        tpu_bdc_fp_reciprocal_v2(out_local_addr[(stage_idx - 1) & 0x1],
                                 in_local_addr[(stage_idx - 1) & 0x1],
                                 BOFFSET(0), BOFFSET(2), &cur_shape, dtype);
      } else if (active_type == ACTIVE_SIGMOID) {
        if (dtype != DT_FP32) {
          _tpu_bdc_fp_sigmoid(out_local_addr[(stage_idx - 1) & 0x1],
                             in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(2),
                             BOFFSET(4), BOFFSET(6), BOFFSET(8),
                             /*coeff = */ BOFFSET(0), /*table = */ BOFFSET(1),
                             &cur_shape, dtype);
        } else {
          // f32 no need cast buffer
          _tpu_bdc_fp_sigmoid(out_local_addr[(stage_idx - 1) & 0x1],
                             in_local_addr[(stage_idx - 1) & 0x1], 0, 0,
                             BOFFSET(2), BOFFSET(3),
                             /*coeff = */ BOFFSET(0), /*table = */ BOFFSET(1),
                             &cur_shape, dtype);
        }
      } else if (active_type == ACTIVE_SILU) {
        if (dtype != DT_FP32) {
          _tpu_bdc_fp_silu(out_local_addr[(stage_idx - 1) & 0x1],
                          in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(2),
                          BOFFSET(4), BOFFSET(6), BOFFSET(8),
                          /*coeff = */ BOFFSET(0), /*table = */ BOFFSET(1),
                          &cur_shape, dtype);
        } else {
          // f32 no need cast buffer
          _tpu_bdc_fp_silu(out_local_addr[(stage_idx - 1) & 0x1],
                          in_local_addr[(stage_idx - 1) & 0x1], 0, 0,
                          BOFFSET(2), BOFFSET(3),
                          /*coeff = */ BOFFSET(0), /*table = */ BOFFSET(1),
                          &cur_shape, dtype);
        }
      } else if (active_type == ACTIVE_ISINF) {
        tpu_bdc_fp_isinf(out_local_addr[(stage_idx - 1) & 0x1],
                         in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(0),
                         &cur_shape, dtype);
      } else if (active_type == ACTIVE_ISNAN) {
        tpu_bdc_fp_isnan(out_local_addr[(stage_idx - 1) & 0x1],
                         in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(0),
                         BOFFSET(1), BOFFSET(2), &cur_shape, dtype);
      } else if (active_type == ACTIVE_RELU) {
        tpu_bdc_relu(out_local_addr[(stage_idx - 1) & 0x1],
                     in_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, dtype);
      } else if (active_type == ACTIVE_EXP2) {
        tpu_bdc_fp_exp2(out_local_addr[(stage_idx - 1) & 0x1],
                        in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(2),
                        BOFFSET(3), BOFFSET(0), BOFFSET(1), &cur_shape, dtype);
      } else if (active_type == ACTIVE_TRUNC) {
        tpu_bdc_fp_trunc(out_local_addr[(stage_idx - 1) & 0x1],
                         in_local_addr[(stage_idx - 1) & 0x1], &cur_shape,
                         dtype);
      }

      else {
        int _cur_shape[4] = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                             w_dim}; // matrix layout shape (n, c, h, w)
        nodechip_active_local(in_local_addr[(stage_idx - 1) & 0x1],
                              out_local_addr[(stage_idx - 1) & 0x1], BOFFSET(0),
                              _cur_shape, dtype, active_type, 0, coef);
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

int tpu_kernel_api_active_multi_core(const void *args) {
  sg_api_active_t *api = (sg_api_active_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
#ifdef ENABLE_MULTI_CORE
  data_type_t out_dtype = (api->active_type == ACTIVE_ISINF ||
                           api->active_type == ACTIVE_ISNAN) ? DT_UINT8 : dtype;
  int output_size = tpu_data_type_size(out_dtype);
  TPUKERNEL_ASSERT(dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16);
  tpu_initialize();
  unsigned int slice_num = tpu_core_num();
  unsigned int slice_idx = tpu_core_index();
  TPUKERNEL_ASSERT(slice_num > 0);
  TPUKERNEL_ASSERT(0 <= slice_idx && slice_idx < slice_num);
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; i++) {
    length *= (unsigned long long)api->shape[i];
  }

  unsigned long long slice = DIV_UP(length, slice_num);
  unsigned int offset = slice_idx * slice;
  unsigned long long real_slice = MIN(slice, length - offset);
  if (offset >= length)
  {
    // No task for me? It's ok.
    return 0;
  }
  const int dsize = tpu_data_type_size(dtype);

  nodechip_active_v2(api->input_global_addr + offset * dsize,
                     api->output_global_addr + offset * output_size, real_slice,
                     dtype, api->active_type, NULL);
  tpu_poll();
  return 0;
#else
  TPUKERNEL_ASSERT(dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16);
  tpu_initialize();
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; i++) {
    length *= (unsigned long long)api->shape[i];
  }

  nodechip_active_v2(api->input_global_addr, api->output_global_addr, length,
                     dtype, api->active_type, NULL);
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_active_multi_core);
