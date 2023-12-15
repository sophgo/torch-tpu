#include "config.h"
#include "sg_api_struct.h"
#include "tpu_kernel.h"

#define BOFFSET(index) buffer_addr + index *tensor_bsize_pnpu

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

void tpu_bdc_pow(local_addr_t dst_addr, local_addr_t src0_addr,
                 local_addr_t src1_addr, local_addr_t dst_fp32_addr,
                 local_addr_t src0_fp32_addr, local_addr_t src1_fp32_addr,
                 local_addr_t work0_addr, local_addr_t work1_addr,
                 local_addr_t exp_coeff_addr, local_addr_t log_coeff_addr,
                 local_addr_t exp_table_addr, const dim4 *shape,
                 data_type_t dst_dtype, data_type_t src0_dtype,
                 data_type_t src1_dtype) {
  if (src0_dtype == DT_FP32) {
    src0_fp32_addr = src0_addr;
  } else {
    tpu_bdc_cast(src0_fp32_addr, src0_addr, shape, NULL, NULL, DT_FP32,
                 src0_dtype, RM_HALF_AWAY_FROM_ZERO);
  }
  if (src1_dtype == DT_FP32) {
    src1_fp32_addr = src1_addr;
  } else {
    tpu_bdc_cast(src1_fp32_addr, src1_addr, shape, NULL, NULL, DT_FP32,
                 src1_dtype, RM_HALF_AWAY_FROM_ZERO);
  }

  if (dst_dtype == DT_FP32) {
    tpu_bdc_fp32_pow(dst_addr, src0_fp32_addr, src1_fp32_addr, work0_addr,
                     work1_addr, exp_coeff_addr, log_coeff_addr, exp_table_addr,
                     shape);
  } else {
    tpu_bdc_fp32_pow(dst_fp32_addr, src0_fp32_addr, src1_fp32_addr, work0_addr,
                     work1_addr, exp_coeff_addr, log_coeff_addr, exp_table_addr,
                     shape);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dst_dtype, DT_FP32,
                 RM_HALF_AWAY_FROM_ZERO);
  }
}

/*
 * output = math.pow(input, other)
 */
void nodechip_pow(global_addr_t output_global_addr,
                  global_addr_t input_global_addr,
                  global_addr_t other_global_addr, unsigned long long length,
                  data_type_t input_dtype, data_type_t other_dtype,
                  data_type_t output_dtype) {

  if (length == 0)
    return;

  unsigned int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
  // 2input1, 2intput2, 2output, 2work, 3coeff
  // int tensor_num = 11;
  // if (input_dtype != DT_FP32) {
  //   tensor_num += 1;
  // }
  // if (other_dtype != DT_FP32) {
  //   tensor_num += 1;
  // }
  // if (output_dtype != DT_FP32) {
  //   tensor_num += 1;
  // }
  // tensor_num > 2*bank_szie, so each tensor can only use one bank

  local_addr_t output_local_addr[2] = {0, 1 * bank_size};
  local_addr_t input_local_addr[2] = {2 * bank_size, 3 * bank_size};
  local_addr_t other_local_addr[2] = {4 * bank_size, 5 * bank_size};
  local_addr_t work0_local_addr = 6 * bank_size;
  local_addr_t work1_local_addr = 7 * bank_size;
  local_addr_t exp_coeff_local_addr = 8 * bank_size;
  local_addr_t log_coeff_local_addr = 9 * bank_size;
  local_addr_t exp_table_local_addr = 10 * bank_size;

  // use fp_local_addr when dtype != DT_FP32
  local_addr_t output_fp_local_addr = 11 * bank_size;
  local_addr_t input_fp_local_addr = 12 * bank_size;
  local_addr_t other_fp_local_addr = 13 * bank_size;

  int dtype_size = tpu_data_type_size(DT_FP32);
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;
  const unsigned int w_dim = tpu_eu_num(DT_FP32);
  const int n_dim = bank_size * tpu_npu_num() / dtype_size / max_m_dim;

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
  while (cur_idx[2] < length) {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1) {
      unsigned long long cur_len = MIN(length - cur_idx[0], n_dim * max_m_dim);
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0];
    }

    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(output_global_addr + cur_idx[2] * dtype_size,
                          output_local_addr[stage_idx & 0x1], cur_n_dim[2],
                          cur_m_dim[2], w_dim, cur_m_dim[2], output_dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(input_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          input_dtype);
      tpu_gdma_matrix_S2L(other_local_addr[stage_idx & 0x1],
                          other_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          other_dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1, w_dim};
      tpu_bdc_pow(output_local_addr[(stage_idx - 1) & 0x1],
                  input_local_addr[(stage_idx - 1) & 0x1],
                  other_local_addr[(stage_idx - 1) & 0x1], output_fp_local_addr,
                  input_fp_local_addr, other_fp_local_addr, work0_local_addr,
                  work1_local_addr, exp_coeff_local_addr, log_coeff_local_addr,
                  exp_table_local_addr, &cur_shape, output_dtype, input_dtype,
                  other_dtype);
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

void tpu_kernel_api_pow(const void *args) {
  sg_api_pow_t *api = (sg_api_pow_t *)args;
  TPUKERNEL_ASSERT(api->input_dtype == DT_FP32 || api->input_dtype == DT_FP16 ||
                   api->input_dtype == DT_BFP16 ||
                   api->input_dtype == DT_INT32);
  TPUKERNEL_ASSERT(api->other_dtype == DT_FP32 || api->other_dtype == DT_FP16 ||
                   api->other_dtype == DT_BFP16 ||
                   api->other_dtype == DT_INT32);

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_pow(api->output_global_addr, api->input_global_addr,
               api->other_global_addr, length, api->input_dtype,
               api->other_dtype, api->output_dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_pow_multi_core(const void *args) {
  sg_api_pow_t *api = (sg_api_pow_t *)args;
  TPUKERNEL_ASSERT(api->input_dtype == DT_FP32 || api->input_dtype == DT_FP16 ||
                   api->input_dtype == DT_BFP16 ||
                   api->input_dtype == DT_INT32);
  TPUKERNEL_ASSERT(api->other_dtype == DT_FP32 || api->other_dtype == DT_FP16 ||
                   api->other_dtype == DT_BFP16 ||
                   api->other_dtype == DT_INT32);
  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
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
  if (core_idx < length_secs) {
    nodechip_pow(
        api->output_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->output_dtype),
        api->input_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->input_dtype),
        api->other_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->other_dtype),
        cur_length_slice, api->input_dtype, api->other_dtype,
        api->output_dtype);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_multi_core);
#endif

// judge two dim can be merged according to their shapes
static inline bool can_be_merged(int a1, int a2, int b1, int b2) {
  // case 0: no more than 65535 after merged
  if (a1 * a2 > 65535 || b1 * b2 > 65535)
    return false;
  // case 1: both dims are same --- always true
  if (a1 * a2 == b1 * b2)
    return true;
  // case 2: only one dim is same --- only when another is 1 can be merged
  if ((a1 == b1 && a2 != b2 && a1 == 1) || (a1 != b1 && a2 == b2 && a2 == 1))
    return true;
  // // case 3: one tensor has two 1
  if ((a1 == a2 && a1 == 1) || (b1 == b2 && b1 == 1))
    return true;
  return false;
}

static inline void merge_two_dims(int *ashape, int *bshape, int dims,
                                  int d_th) {
  ashape[d_th] *= ashape[d_th + 1];
  bshape[d_th] *= bshape[d_th + 1];
  for (int i = d_th + 1; i < dims - 1; i++) {
    ashape[i] = ashape[i + 1];
    bshape[i] = bshape[i + 1];
  }
}

static inline void split_cw(int v, int *c, int *w) {
  // Split v into c and w
  *c = 1;
  *w = v;
  for (int i = NPU_NUM; i >= 2; i--) {
    if (v % i == 0) {
      *c = i;
      *w = v / i;
      return;
    }
  }
}

static inline void assign_shapes(int *shape, int dim) {
  if (dim >= 4) {
    return;
  } else if (dim == 1) {
    split_cw(shape[0], &shape[1], &shape[3]);
    shape[0] = 1;
    shape[2] = 1;
    return;
  } else if (dim == 2) {
    // only use c and w
    shape[3] = shape[1];
    shape[1] = shape[0];
    shape[2] = 1;
    shape[0] = 1;
    return;
  } else if (dim == 3) {
    // use c,h,w
    shape[3] = shape[2];
    shape[2] = shape[1];
    shape[1] = shape[0];
    shape[0] = 1;
    return;
  }
}

void nodechip_pow_bcast(global_addr_t output_global_addr,
                        global_addr_t input_global_addr,
                        global_addr_t other_global_addr, dim4 *output_shape,
                        dim4 *input_shape, dim4 *other_shape,
                        dim4 *output_global_stride, dim4 *input_global_stride,
                        dim4 *other_global_stride, data_type_t output_dtype,
                        data_type_t input_dtype, data_type_t other_dtype) {

  bool input_bcast[4] = {
      input_shape->n != output_shape->n, input_shape->c != output_shape->c,
      input_shape->h != output_shape->h, input_shape->w != output_shape->w};
  bool other_bcast[4] = {
      other_shape->n != output_shape->n, other_shape->c != output_shape->c,
      other_shape->h != output_shape->h, other_shape->w != output_shape->w};

  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
  // tensor_num > 2*bank_szie, so each tensor can only use one bank
  local_addr_t output_local_addr[2] = {0, 1 * bank_size};
  local_addr_t input_local_addr[2] = {2 * bank_size, 3 * bank_size};
  local_addr_t other_local_addr[2] = {4 * bank_size, 5 * bank_size};
  local_addr_t work0_local_addr = 6 * bank_size;
  local_addr_t work1_local_addr = 7 * bank_size;
  local_addr_t exp_coeff_local_addr = 8 * bank_size;
  local_addr_t log_coeff_local_addr = 9 * bank_size;
  local_addr_t exp_table_local_addr = 10 * bank_size;

  // use fp_local_addr when dtype != DT_FP32
  local_addr_t output_fp_local_addr = 11 * bank_size;
  local_addr_t input_fp_local_addr = 12 * bank_size;
  local_addr_t other_fp_local_addr = 13 * bank_size;

  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);

  const int c_per_npu = DIV_UP(output_shape->c, NPU_NUM);
  const int eu_num = tpu_eu_num(DT_FP32);
  int nmax = output_shape->n, hmax = output_shape->h,
      cmax = c_per_npu * NPU_NUM,
      wmax = MIN(output_shape->w, tpu_gdma_shape_limit(TENSOR_W_DIM));

  while (true) {
    int size = tpu_aligned_feature_size(hmax, wmax, DT_FP32) *
               DIV_UP(cmax, NPU_NUM) * nmax;
    if (size <= bank_size) {
      break;
    } else {
      if (cmax > NPU_NUM) {
        cmax -= (cmax % NPU_NUM == 0) ? NPU_NUM : cmax % NPU_NUM;
        continue;
      } else if (nmax > 1) {
        nmax /= 2;
        continue;
      } else if (hmax > 1) {
        hmax /= 2;
        continue;
      } else if (wmax > eu_num) {
        wmax -= (wmax % eu_num == 0) ? eu_num : wmax % eu_num;
        continue;
      } else {
        TPUKERNEL_ASSERT(false);
      }
    }
  }
  int index = 0;
  bool l2s = false;
  global_addr_t output_global_addr_gdma = 0;
  dim4 gdma_shape;

  // dim4 slice_shape = {.w = output_shape->w};
  dim4 slice_shape;
  dim4 input_local_shape, other_local_shape;
  dim4 input_local_stride, other_local_stride;
  int ctodo = output_shape->c, cdone = 0;
  while (ctodo > 0) {
    slice_shape.c = MIN(ctodo, cmax);
    int ntodo = output_shape->n, ndone = 0;
    while (ntodo > 0) {
      slice_shape.n = MIN(ntodo, nmax);
      int htodo = output_shape->h, hdone = 0;
      while (htodo > 0) {
        slice_shape.h = MIN(htodo, hmax);
        int wtodo = output_shape->w, wdone = 0;
        while (wtodo > 0) {
          slice_shape.w = MIN(wtodo, wmax);

          // Move input from global memory to local memory
          tpu_aligned_stride(&input_local_stride, 0, &slice_shape, input_dtype);
          input_local_shape.n = input_bcast[0] ? 1 : slice_shape.n;
          input_local_shape.c = input_bcast[1] ? 1 : slice_shape.c;
          input_local_shape.h = input_bcast[2] ? 1 : slice_shape.h;
          input_local_shape.w = input_bcast[3] ? 1 : slice_shape.w;
          global_addr_t input_global_addr_gdma =
              input_global_addr +
              ((input_bcast[0] ? 0 : ndone) * input_global_stride->n +
               (input_bcast[1] ? 0 : cdone) * input_global_stride->c +
               (input_bcast[2] ? 0 : hdone) * input_global_stride->h +
               (input_bcast[3] ? 0 : wdone) * input_global_stride->w) *
                  tpu_data_type_size(input_dtype);
          tpu_gdma_cpy_S2L(input_local_addr[index], input_global_addr_gdma,
                           &input_local_shape, &input_local_stride,
                           input_global_stride, input_dtype);
          // Move other from global memory to local memory
          tpu_aligned_stride(&other_local_stride, 0, &slice_shape, other_dtype);
          other_local_shape.n = other_bcast[0] ? 1 : slice_shape.n;
          other_local_shape.c = other_bcast[1] ? 1 : slice_shape.c;
          other_local_shape.h = other_bcast[2] ? 1 : slice_shape.h;
          other_local_shape.w = other_bcast[3] ? 1 : slice_shape.w;
          global_addr_t other_global_addr_gdma =
              other_global_addr +
              ((other_bcast[0] ? 0 : ndone) * other_global_stride->n +
               (other_bcast[1] ? 0 : cdone) * other_global_stride->c +
               (other_bcast[2] ? 0 : hdone) * other_global_stride->h +
               (other_bcast[3] ? 0 : wdone) * other_global_stride->w) *
                  tpu_data_type_size(other_dtype);
          tpu_gdma_cpy_S2L(other_local_addr[index], other_global_addr_gdma,
                           &other_local_shape, &other_local_stride,
                           other_global_stride, other_dtype);

          // Broadcast input if needed
          if (input_bcast[1]) {
            input_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast(input_local_addr[index], input_local_addr[index],
                              &input_local_shape, input_dtype);
          }
          if (input_bcast[0] || input_bcast[2] || input_bcast[3] ||
              (input_bcast[1] && slice_shape.c > NPU_NUM)) {
            for (int i = 0; i < 4; ++i) {
              ((int *)&input_local_stride)[i] =
                  input_bcast[i] ? 0 : ((int *)&input_local_stride)[i];
            }
            tpu_bdc_cpy(input_local_addr[index], input_local_addr[index],
                        &slice_shape, NULL, &input_local_stride, input_dtype);
          }

          // Broadcast other if needed
          if (other_bcast[1]) {
            other_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast(other_local_addr[index], other_local_addr[index],
                              &other_local_shape, other_dtype);
          }
          if (other_bcast[0] || other_bcast[2] || other_bcast[3] ||
              (other_bcast[1] && slice_shape.c > NPU_NUM)) {
            for (int i = 0; i < 4; ++i) {
              ((int *)&other_local_stride)[i] =
                  other_bcast[i] ? 0 : ((int *)&other_local_stride)[i];
            }
            tpu_bdc_cpy(other_local_addr[index], other_local_addr[index],
                        &slice_shape, NULL, &other_local_stride, other_dtype);
          }

          tpu_parallel_start();

          if (l2s) {
            // Move out from local memory to global memory
            tpu_gdma_cpy_L2S(output_global_addr_gdma,
                             output_local_addr[1 - index], &gdma_shape,
                             output_global_stride, NULL, output_dtype);
          }

          tpu_bdc_pow(output_local_addr[index], input_local_addr[index],
                      other_local_addr[index], output_fp_local_addr,
                      input_fp_local_addr, other_fp_local_addr,
                      work0_local_addr, work1_local_addr, exp_coeff_local_addr,
                      log_coeff_local_addr, exp_table_local_addr, &slice_shape,
                      output_dtype, input_dtype, other_dtype);

          tpu_parallel_end();

          output_global_addr_gdma =
              output_global_addr + (ndone * output_global_stride->n +
                                    cdone * output_global_stride->c +
                                    hdone * output_global_stride->h +
                                    wdone * output_global_stride->w) *
                                       tpu_data_type_size(output_dtype);
          gdma_shape = slice_shape;
          l2s = true;
          index = 1 - index;
          wtodo -= slice_shape.w;
          wdone += slice_shape.w;
        }
        htodo -= slice_shape.h;
        hdone += slice_shape.h;
      }

      ntodo -= slice_shape.n;
      ndone += slice_shape.n;
    }
    ctodo -= slice_shape.c;
    cdone += slice_shape.c;
  }

  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }
  if (l2s) {
    // Move out from local memory to global memory
    tpu_gdma_cpy_L2S(output_global_addr_gdma, output_local_addr[1 - index],
                     &gdma_shape, output_global_stride, NULL, output_dtype);
  }
}

void tpu_kernel_api_pow_bcast(const void *args) {
  sg_api_pow_bcast_t *api = (sg_api_pow_bcast_t *)args;
  TPUKERNEL_ASSERT(api->input_dtype == DT_FP32 || api->input_dtype == DT_FP16 ||
                   api->input_dtype == DT_BFP16 ||
                   api->input_dtype == DT_INT32);
  TPUKERNEL_ASSERT(api->other_dtype == DT_FP32 || api->other_dtype == DT_FP16 ||
                   api->other_dtype == DT_BFP16 ||
                   api->other_dtype == DT_INT32);

  // assign n-shape to 4-shape
  int dim = api->dim;
  if (dim < 4) {
    assign_shapes(api->input_shape, dim);
    assign_shapes(api->other_shape, dim);
    dim = 4;
  } else if (dim > 4) {
    int i = 0;
    while (i < dim - 1) {
      if (can_be_merged(api->input_shape[i], api->input_shape[i + 1],
                        api->other_shape[i], api->other_shape[i + 1])) {
        merge_two_dims(api->input_shape, api->other_shape, dim, i);
        dim--;
      } else {
        ++i;
      }
      if (dim == 4) {
        break;
      }
    }
  }

  TPUKERNEL_ASSERT_INFO(dim == 4, "can merge %d dims to 4 dims", dim);

  dim4 input_shape, other_shape, output_shape;
  for (int i = 0; i < 4; i++) {
    ((int *)&input_shape)[i] = api->input_shape[i];
    ((int *)&other_shape)[i] = api->other_shape[i];
    ((int *)&output_shape)[i] = MAX(api->input_shape[i], api->other_shape[i]);
  }

  dim4 input_global_stride, other_global_stride, output_global_stride;
  tpu_continuous_stride(&input_global_stride, &input_shape);
  tpu_continuous_stride(&other_global_stride, &other_shape);
  tpu_continuous_stride(&output_global_stride, &output_shape);

  tpu_initialize();
  nodechip_pow_bcast(api->output_global_addr, api->input_global_addr,
                     api->other_global_addr, &output_shape, &input_shape,
                     &other_shape, &output_global_stride, &input_global_stride,
                     &other_global_stride, api->output_dtype, api->input_dtype,
                     api->other_dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_bcast);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_pow_bcast_multi_core(const void *args) {
  sg_api_pow_bcast_t *api = (sg_api_pow_bcast_t *)args;
  TPUKERNEL_ASSERT(api->input_dtype == DT_FP32 || api->input_dtype == DT_FP16 ||
                   api->input_dtype == DT_BFP16 ||
                   api->input_dtype == DT_INT32);
  TPUKERNEL_ASSERT(api->other_dtype == DT_FP32 || api->other_dtype == DT_FP16 ||
                   api->other_dtype == DT_BFP16 ||
                   api->other_dtype == DT_INT32);

  // assign n-shape to 4-shape
  int dim = api->dim;
  if (dim < 4) {
    assign_shapes(api->input_shape, dim);
    assign_shapes(api->other_shape, dim);
    dim = 4;
  } else if (dim > 4) {
    int i = 0;
    while (i < dim - 1) {
      if (can_be_merged(api->input_shape[i], api->input_shape[i + 1],
                        api->other_shape[i], api->other_shape[i + 1])) {
        merge_two_dims(api->input_shape, api->other_shape, dim, i);
        dim--;
      } else {
        ++i;
      }
      if (dim == 4) {
        break;
      }
    }
  }

  TPUKERNEL_ASSERT_INFO(dim == 4, "can merge %d dims to 4 dims", dim);
  dim4 input_shape, other_shape, output_shape;
  for (int i = 0; i < 4; i++) {
    ((int *)&input_shape)[i] = api->input_shape[i];
    ((int *)&other_shape)[i] = api->other_shape[i];
    ((int *)&output_shape)[i] = MAX(api->input_shape[i], api->other_shape[i]);
  }

  const int core_num = tpu_core_num();
  const int core_idx = tpu_core_index();
  // Select parallel dimensions
  int split_dim = -1;

  for (int i = 0; i < dim; ++i) {
    if (((int *)&input_shape)[i] == ((int *)&other_shape)[i] &&
        ((int *)&input_shape)[i] >= core_num) {
      split_dim = i;
      break;
    }
  }
  if (split_dim == -1) {
    for (int i = 0; i < dim; ++i) {
      if (((int *)&input_shape)[i] == ((int *)&other_shape)[i] &&
          ((int *)&input_shape)[i] != 1) {
        split_dim = i;
        break;
      }
    }
  }
  if (split_dim == -1) {
    split_dim = dim - 1;
  }

  bool input_is_one = ((int *)&input_shape)[split_dim] == 1,
       other_is_one = ((int *)&other_shape)[split_dim] == 1;

  int length = input_is_one ? ((int *)&other_shape)[split_dim]
                            : ((int *)&input_shape)[split_dim];
  int length_slice = DIV_UP(length, core_num);
  int allocate_core = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(allocate_core <= core_num);

  int last_length_slice = length - length_slice * (allocate_core - 1);

  dim4 input_global_stride = {0}, other_global_stride = {0},
       output_global_stride = {0};
  tpu_continuous_stride(&input_global_stride, &input_shape);
  tpu_continuous_stride(&other_global_stride, &other_shape);
  tpu_continuous_stride(&output_global_stride, &output_shape);

  // mutil core
  ((int *)&input_shape)[split_dim] =
      input_is_one
          ? 1
          : (core_idx == allocate_core - 1 ? last_length_slice : length_slice);
  ((int *)&other_shape)[split_dim] =
      other_is_one
          ? 1
          : (core_idx == allocate_core - 1 ? last_length_slice : length_slice);
  ((int *)&output_shape)[split_dim] =
      (input_is_one && other_is_one)
          ? 1
          : (core_idx == allocate_core - 1 ? last_length_slice : length_slice);

  size_t input_offset =
      input_is_one ? 0
                   : ((int *)&input_global_stride)[split_dim] *
                         tpu_data_type_size((data_type_t)api->input_dtype);
  size_t other_offset =
      other_is_one ? 0
                   : ((int *)&other_global_stride)[split_dim] *
                         tpu_data_type_size((data_type_t)api->other_dtype);
  size_t output_offset = ((int *)&output_global_stride)[split_dim] *
                         tpu_data_type_size((data_type_t)api->output_dtype);

  tpu_initialize();
  if (core_idx < allocate_core) {
    nodechip_pow_bcast(
        api->output_global_addr + (length_slice * core_idx) * output_offset,
        api->input_global_addr + (length_slice * core_idx) * input_offset,
        api->other_global_addr + (length_slice * core_idx) * other_offset,
        &output_shape, &input_shape, &other_shape, &output_global_stride,
        &input_global_stride, &other_global_stride, api->output_dtype,
        api->input_dtype, api->other_dtype);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_bcast_multi_core);
#endif

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

void tpu_bdc_int_pow_234(local_addr_t dst_addr, local_addr_t src_addr,
                         local_addr_t buffer1_addr, local_addr_t buffer2_addr,
                         const dim4 *shape, float C, data_type_t dtype,
                         bool out_is_int) {
  if (out_is_int) {
    buffer2_addr = dst_addr;
  }
  if (C == 2) {
    buffer1_addr = buffer2_addr;
  }
  tpu_bdc_int_mul(buffer1_addr, src_addr, src_addr, shape, NULL, NULL, NULL,
                  dtype, dtype, dtype, 0, NO_USE, false);

  if (C == 4.0) {
    tpu_bdc_int_mul(buffer2_addr, buffer1_addr, buffer1_addr, shape, NULL, NULL,
                    NULL, dtype, dtype, dtype, 0, NO_USE, false);
  } else if (C == 3.0) {
    tpu_bdc_int_mul(buffer2_addr, buffer1_addr, src_addr, shape, NULL, NULL,
                    NULL, dtype, dtype, dtype, 0, NO_USE, false);
  }
  if (!out_is_int) {
    tpu_bdc_cast(dst_addr, buffer2_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_AWAY_FROM_ZERO);
  }
}

// (2+2) + 2  = 6
// (2+2) + 2 + 2 + 2 + 2 = 12
void tpu_bdc_pow_C(local_addr_t dst_addr, local_addr_t src_addr,
                   local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                   local_addr_t work0_addr, local_addr_t work1_addr,
                   local_addr_t exp_coeff_addr, local_addr_t log_coeff_addr,
                   local_addr_t exp_table_addr, float C, const dim4 *shape,
                   data_type_t dtype, bool out_is_int) {
  if (dtype == DT_FP32) {
    tpu_bdc_fp32_pow_C(dst_addr, src_addr, work0_addr, work1_addr,
                       exp_coeff_addr, log_coeff_addr, exp_table_addr, C,
                       shape);
  } else if (dtype == DT_INT32) {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_AWAY_FROM_ZERO);
    if (out_is_int) {
      tpu_bdc_fp32_pow_C(dst_fp32_addr, src_fp32_addr, work0_addr, work1_addr,
                         exp_coeff_addr, log_coeff_addr, exp_table_addr, C,
                         shape);
      tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                   RM_HALF_AWAY_FROM_ZERO);
    } else {
      tpu_bdc_fp32_pow_C(dst_addr, src_fp32_addr, work0_addr, work1_addr,
                         exp_coeff_addr, log_coeff_addr, exp_table_addr, C,
                         shape);
    }
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
                                           data_type_t dtype, bool out_is_int) {

  if (length == 0) {
    return;
  }

  int bank_num = tpu_bank_num();

  int tensor_num = 2 + 2; //  2 inputs, 2 outputs
  bool pow_234 = false;
  if (C == 2 || C == 3 || C == 4) {
    tensor_num += out_is_int ? 1 : 2;
    pow_234 = true;
  } else if (dtype == DT_FP32) {
    // work0, work1
    tensor_num += 2;
    // 3bank for 3coeff
    bank_num -= 3;
  } else if (dtype == DT_INT32) {
    // work0, work1, dst_fp32, src_fp32
    tensor_num += (2 + 2);
    // 3bank for 3coeff
    bank_num -= 3;
  } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
    // dst_fp32, src_fp32, f32_work0, f32_work1, fp32 need 2 fp16
    tensor_num += (2 + 2 + 2 + 2);
    // 3bank for 3coeff
    bank_num -= 3;
  } else {
    TPUKERNEL_ASSERT(false);
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_bsize_pnpu = bank_num / tensor_num * bank_bsize;
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

  local_addr_t exp_coeff_addr = tensor_num * tensor_bsize_pnpu;
  local_addr_t log_coeff_addr = exp_coeff_addr + bank_bsize;
  local_addr_t exp_table_addr = log_coeff_addr + bank_bsize;

  if (!pow_234) {
    tpu_bdc_load_fp32_exp_coeff(exp_coeff_addr);
    tpu_bdc_load_fp32_log_coeff(log_coeff_addr);
    tpu_bdc_load_fp32_exp_table(exp_table_addr);
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
        if (dtype == DT_INT32) {
          tpu_bdc_int_pow_234(out_local_addr[(stage_idx - 1) & 0x1],
                              in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(0),
                              BOFFSET(1), &cur_shape, C, dtype, out_is_int);
        } else {
          tpu_bdc_fp_pow_234(out_local_addr[(stage_idx - 1) & 0x1],
                             in_local_addr[(stage_idx - 1) & 0x1], buffer_addr,
                             &cur_shape, C, dtype);
        }

      } else if (dtype == DT_FP32) {
        tpu_bdc_pow_C(out_local_addr[(stage_idx - 1) & 0x1],
                      in_local_addr[(stage_idx - 1) & 0x1], 0, 0, BOFFSET(0),
                      BOFFSET(1), exp_coeff_addr, log_coeff_addr,
                      exp_table_addr, C, &cur_shape, dtype, out_is_int);
      } else if (dtype == DT_INT32) {
        tpu_bdc_pow_C(out_local_addr[(stage_idx - 1) & 0x1],
                      in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(2),
                      BOFFSET(3), BOFFSET(0), BOFFSET(1), exp_coeff_addr,
                      log_coeff_addr, exp_table_addr, C, &cur_shape, dtype,
                      out_is_int);
      } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
        tpu_bdc_pow_C(out_local_addr[(stage_idx - 1) & 0x1],
                      in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(4),
                      BOFFSET(6), BOFFSET(0), BOFFSET(2), exp_coeff_addr,
                      log_coeff_addr, exp_table_addr, C, &cur_shape, dtype,
                      out_is_int);
      }
      /*
      pow_234:
        DT_INT32
          0: buffer1
          1: buffer2
        ELSE
          0: buffer
      DT_FP32:
        0: work0
        1: work1
        exp_coeff, log_coeff, exp_table
      DT_INT32:
        0: work0
        1: work1
        2: dst_fp32
        3: src_fp32
        exp_coeff, log_coeff, exp_table
      DT_FP16:
        0-1: work0
        2-3: work1
        4-5: dst_fp32
        6-7: src_fp32
        exp_coeff, log_coeff, exp_table
      */
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
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_pow_c_parallel(api->self_global_addr, api->out_global_addr, length,
                          api->value, api->dtype, api->out_is_int);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_c);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_pow_c_multi_core(const void *args) {
  sg_api_pow_tensor_scalar_t *api = (sg_api_pow_tensor_scalar_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  int type_size = tpu_data_type_size(api->dtype);

  tpu_initialize();
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  if (core_idx < length_secs) {
    nodechip_pow_c_parallel(
        api->self_global_addr + (length_slice * core_idx) * type_size,
        api->out_global_addr + (length_slice * core_idx) * type_size,
        cur_length_slice, api->value, api->dtype, api->out_is_int);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pow_c_multi_core);
#endif

void tpu_bdc_C_pow(local_addr_t dst_addr, local_addr_t src_addr,
                   local_addr_t dst_fp32_addr, local_addr_t src_fp32_addr,
                   local_addr_t work0_addr, local_addr_t work1_addr,
                   local_addr_t exp_coeff_addr, local_addr_t exp_table_addr,
                   float C, const dim4 *shape, data_type_t dtype,
                   bool out_is_int) {
  if (dtype == DT_FP32) {
    tpu_bdc_fp32_C_pow(dst_addr, src_addr, work0_addr, work1_addr,
                       exp_coeff_addr, exp_table_addr, C, shape);
  } else if (dtype == DT_INT32) {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_AWAY_FROM_ZERO);
    if (out_is_int) {
      tpu_bdc_fp32_C_pow(dst_fp32_addr, src_fp32_addr, work0_addr, work1_addr,
                         exp_coeff_addr, exp_table_addr, C, shape);
      tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                   RM_HALF_AWAY_FROM_ZERO);
    } else {
      tpu_bdc_fp32_C_pow(dst_addr, src_fp32_addr, work0_addr, work1_addr,
                         exp_coeff_addr, exp_table_addr, C, shape);
    }
  } else {
    tpu_bdc_cast(src_fp32_addr, src_addr, shape, NULL, NULL, DT_FP32, dtype,
                 RM_HALF_AWAY_FROM_ZERO);
    tpu_bdc_fp32_C_pow(dst_fp32_addr, src_fp32_addr, work0_addr, work1_addr,
                       exp_coeff_addr, exp_table_addr, C, shape);
    tpu_bdc_cast(dst_addr, dst_fp32_addr, shape, NULL, NULL, dtype, DT_FP32,
                 RM_HALF_AWAY_FROM_ZERO);
  }
}

static inline void nodechip_c_pow_parallel(global_addr_t in_global_addr,
                                           global_addr_t out_global_addr,
                                           unsigned long long length, float C,
                                           data_type_t dtype, bool out_is_int) {

  if (length == 0) {
    return;
  }

  // 2bank for coeff
  int bank_num = tpu_bank_num() - 2;

  int tensor_num = 2 + 2; //  2 inputs, 2 outputs

  if (dtype == DT_FP32) {
    // work0, work1
    tensor_num += 2;
  } else if (dtype == DT_INT32) {
    // work0, work1, dst_fp32, src_fp32
    tensor_num += (2 + 2);
  } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
    // dst_fp32, src_fp32, f32_work0, f32_work1, fp32 need 2 fp16
    tensor_num += (2 + 2 + 2 + 2);
  } else {
    TPUKERNEL_ASSERT(false);
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_npu_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_bsize_pnpu = bank_num / tensor_num * bank_bsize;
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

  local_addr_t exp_coeff_addr = tensor_num * tensor_bsize_pnpu;
  local_addr_t exp_table_addr = exp_coeff_addr + bank_bsize;

  tpu_bdc_load_fp32_exp_coeff(exp_coeff_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_addr);

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

      if (dtype == DT_FP32) {
        tpu_bdc_C_pow(out_local_addr[(stage_idx - 1) & 0x1],
                      in_local_addr[(stage_idx - 1) & 0x1], 0, 0, BOFFSET(0),
                      BOFFSET(1), exp_coeff_addr, exp_table_addr, C, &cur_shape,
                      dtype, out_is_int);
      } else if (dtype == DT_INT32) {
        tpu_bdc_C_pow(out_local_addr[(stage_idx - 1) & 0x1],
                      in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(2),
                      BOFFSET(3), BOFFSET(0), BOFFSET(1), exp_coeff_addr,
                      exp_table_addr, C, &cur_shape, dtype, out_is_int);
      } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
        tpu_bdc_C_pow(out_local_addr[(stage_idx - 1) & 0x1],
                      in_local_addr[(stage_idx - 1) & 0x1], BOFFSET(4),
                      BOFFSET(6), BOFFSET(0), BOFFSET(2), exp_coeff_addr,
                      exp_table_addr, C, &cur_shape, dtype, out_is_int);
      }
      /*
      DT_FP32:
        0: work0
        1: work1
        exp_coeff, exp_table
      DT_INT32:
        0: work0
        1: work1
        2: dst_fp32
        3: src_fp32
      DT_FP16:
        0-1: work0
        2-3: work1
        4-5: dst_fp32
        6-7: src_fp32
        exp_coeff, exp_table
      */
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

void tpu_kernel_api_c_pow(const void *args) {
  sg_api_pow_tensor_scalar_t *api = (sg_api_pow_tensor_scalar_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_c_pow_parallel(api->self_global_addr, api->out_global_addr, length,
                          api->value, api->dtype, api->out_is_int);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_c_pow);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_c_pow_multi_core(const void *args) {
  sg_api_pow_tensor_scalar_t *api = (sg_api_pow_tensor_scalar_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  int type_size = tpu_data_type_size(api->dtype);

  tpu_initialize();
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  if (core_idx < length_secs) {
    nodechip_c_pow_parallel(
        api->self_global_addr + (length_slice * core_idx) * type_size,
        api->out_global_addr + (length_slice * core_idx) * type_size,
        cur_length_slice, api->value, api->dtype, api->out_is_int);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_c_pow_multi_core);
#endif