
#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = input + value * ( other )
 */
typedef void (*const_binary_int_func)(local_addr_t, local_addr_t, scalar_t,
                                      const dim4 *, const dim4 *, const dim4 *,
                                      data_type_t, data_type_t, data_type_t,
                                      char, rounding_mode_t, bool);

typedef void (*const_binary_fp_func)(local_addr_t, local_addr_t, scalar_t,
                                     const dim4 *, const dim4 *, const dim4 *,
                                     data_type_t);

typedef void (*binary_int_func)(local_addr_t, local_addr_t, local_addr_t,
                                const dim4 *, const dim4 *, const dim4 *,
                                const dim4 *, data_type_t, data_type_t,
                                data_type_t, char, rounding_mode_t, bool);

typedef void (*binary_fp_func)(local_addr_t, local_addr_t, local_addr_t,
                               const dim4 *, const dim4 *, const dim4 *,
                               const dim4 *, data_type_t);

typedef void (*const_binary_fp32_div_func)(local_addr_t, local_addr_t, float,
                                           const dim4 *, const dim4 *,
                                           const dim4 *);

static inline const_binary_int_func get_const_binary_int_func(int binary_type,
                                                              bool inversed) {
  const_binary_int_func func = NULL;
  if (binary_type == BINARY_ADD)
    func = tpu_bdc_int_add_C;
  else if (binary_type == BINARY_MUL)
    func = tpu_bdc_int_mul_C;
  else if (binary_type == BINARY_SUB && !inversed)
    func = tpu_bdc_int_sub_C;
  else if (binary_type == BINARY_SUB && inversed)
    func = tpu_bdc_int_C_sub;
  return func;
}

static inline const_binary_fp_func get_const_binary_fp_func(int binary_type,
                                                            bool inversed) {
  const_binary_fp_func func = NULL;
  if (binary_type == BINARY_ADD)
    func = tpu_bdc_fp_add_C;
  else if (binary_type == BINARY_MUL)
    func = tpu_bdc_fp_mul_C;
  else if (binary_type == BINARY_SUB && !inversed)
    func = tpu_bdc_fp_sub_C;
  else if (binary_type == BINARY_SUB && inversed)
    func = tpu_bdc_fp_C_sub;
  return func;
}

static inline binary_int_func get_binary_int_func(int binary_type) {
  binary_int_func func = NULL;
  if (binary_type == BINARY_ADD)
    func = tpu_bdc_int_add;
  else if (binary_type == BINARY_SUB)
    func = tpu_bdc_int_sub;
  else if (binary_type == BINARY_MUL)
    func = tpu_bdc_int_mul;
  return func;
}

static inline binary_fp_func get_binary_fp_func(int binary_type) {
  binary_fp_func func = NULL;
  if (binary_type == BINARY_ADD)
    func = tpu_bdc_fp_add;
  else if (binary_type == BINARY_SUB)
    func = tpu_bdc_fp_sub;
  else if (binary_type == BINARY_MUL)
    func = tpu_bdc_fp_mul;
  else if (binary_type == BINARY_DIV)
    func = tpu_bdc_fp_div;
  return func;
}

static inline const_binary_fp32_div_func
get_const_binary_fp32_div_func(bool inversed) {
  const_binary_fp32_div_func func = NULL;
  func = inversed ? tpu_bdc_fp32_C_div : tpu_bdc_fp32_div_C;
  return func;
}

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

void nodechip_binary_div(global_addr_t input_global_addr,
                         global_addr_t other_global_addr,
                         global_addr_t output_global_addr,
                         unsigned long long length, data_type_t dtype) {
  if (length == 0) {
    return;
  }

  int input_size = 0, output_size = 0, buffer_size = 0;
  int buffer_num = 2;
  data_type_t save_dtype = DT_FP32;
  if (dtype == DT_FP32) {
    // 4*input + 2*output + 0*buffer = 12 bank
    input_size = 2 * BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 0;
    buffer_num = 0;
  } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
    // 4*input + 2*output + 3*buffer = 12 bank
    input_size = BANK_SIZE;
    output_size = BANK_SIZE;
    buffer_size = 2 * BANK_SIZE;
    buffer_num = 3;
    save_dtype = dtype;
    // #ifdef BACKEND_SG2260
    //     input_size = 2 * BANK_SIZE;
    //     output_size = 2 * BANK_SIZE;
    //     buffer_size = 0;
    //     buffer_num = 0;
    // #endif
  } else if (dtype == DT_INT32) {
    // 4*input + 2*output + 2*buffer = 16 bank
    input_size = 2 * BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 2 * BANK_SIZE;
  } else if (dtype == DT_INT16 || dtype == DT_UINT16) {
    // 4*input + 2*output + 2*buffer = 12 bank
    input_size = BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 2 * BANK_SIZE;
  } else if (dtype == DT_INT8 || dtype == DT_UINT8) {
    // 4*input + 2*output + 2*buffer = 16 bank
    input_size = BANK_SIZE;
    output_size = 3 * BANK_SIZE;
    buffer_size = 3 * BANK_SIZE;
  }

  TPUKERNEL_ASSERT(input_size * 4 + output_size * 2 +
                       buffer_size * buffer_num <=
                   LOCAL_MEM_SIZE);

  int dtype_size = tpu_data_type_size(dtype);
  int load_size = (dtype_size == 1) ? input_size * 3 / 4 : input_size;

  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;
  const unsigned int w_dim = tpu_eu_num(DT_FP32);
  const int n_dim = load_size * tpu_npu_num() / dtype_size / max_m_dim;

  local_addr_t input_local_addr[2] = {0, 1 * input_size};
  local_addr_t other_local_addr[2] = {2 * input_size, 3 * input_size};
  local_addr_t output_local_addr[2] = {4 * input_size,
                                       4 * input_size + output_size};
  local_addr_t buffer_start_addr = output_local_addr[1] + output_size;
  local_addr_t buffer_fp32_local_addr[3] = {
      buffer_start_addr, buffer_start_addr + buffer_size,
      buffer_start_addr + 2 * buffer_size};

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

      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0]; // cur_len / cur_m_dim[0] >= 1
    }

    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(
          output_global_addr + cur_idx[2] * tpu_data_type_size(save_dtype),
          output_local_addr[stage_idx & 0x1],
          /**rows, cols, cols_per_channel, row_stride*/
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2], save_dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(input_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);

      tpu_gdma_matrix_S2L(other_local_addr[stage_idx & 0x1],
                          other_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // calculate
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)

      if (dtype == DT_FP32) {
        tpu_bdc_fp_div(output_local_addr[(stage_idx - 1) & 0x1],
                       input_local_addr[(stage_idx - 1) & 0x1],
                       other_local_addr[(stage_idx - 1) & 0x1], &cur_shape,
                       NULL, NULL, NULL, DT_FP32);
      } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
        // #ifdef BACKEND_1684X
        tpu_bdc_cast(buffer_fp32_local_addr[0],
                     input_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        tpu_bdc_cast(buffer_fp32_local_addr[1],
                     other_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        tpu_bdc_fp_div(buffer_fp32_local_addr[2], buffer_fp32_local_addr[0],
                       buffer_fp32_local_addr[1], &cur_shape, NULL, NULL, NULL,
                       DT_FP32);
        tpu_bdc_cast(output_local_addr[(stage_idx - 1) & 0x1],
                     buffer_fp32_local_addr[2], &cur_shape, NULL, NULL, dtype,
                     DT_FP32, RM_HALF_TO_EVEN);
        // #endif
        // #ifdef BACKEND_SG2260
        //         tpu_bdc_fp_div(output_local_addr[(stage_idx - 1) & 0x1],
        //                        input_local_addr[(stage_idx - 1) & 0x1],
        //                        other_local_addr[(stage_idx - 1) & 0x1],
        //                        &cur_shape, NULL, NULL, NULL, dtype);
        // #endif
      } else {
        tpu_bdc_cast(buffer_fp32_local_addr[0],
                     input_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        tpu_bdc_cast(buffer_fp32_local_addr[1],
                     other_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        tpu_bdc_fp_div(output_local_addr[(stage_idx - 1) & 0x1],
                       buffer_fp32_local_addr[0], buffer_fp32_local_addr[1],
                       &cur_shape, NULL, NULL, NULL, DT_FP32);
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

void nodechip_binary(global_addr_t input_global_addr,
                     global_addr_t other_global_addr,
                     global_addr_t output_global_addr, scalar_t value,
                     unsigned long long length, data_type_t dtype,
                     sg_binary_type_t binary_type) {
  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_num = 2 + 2 + 2;
  if (binary_type == BINARY_ADD || binary_type == BINARY_SUB) {
    tensor_num += 1;
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
  local_addr_t other_local_addr[2] = {2 * tensor_bsize_pnpu,
                                      3 * tensor_bsize_pnpu};
  local_addr_t out_local_addr[2] = {4 * tensor_bsize_pnpu,
                                    5 * tensor_bsize_pnpu};

  local_addr_t buffer_local_addr = 6 * tensor_bsize_pnpu;

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
      tpu_gdma_matrix_L2S(output_global_addr + cur_idx[2] * dtype_size,
                          out_local_addr[stage_idx & 0x1],
                          /**rows, cols, cols_per_channel, row_stride*/
                          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],
                          dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(in_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);

      tpu_gdma_matrix_S2L(other_local_addr[stage_idx & 0x1],
                          other_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // calculate
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)
      if (dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16) {
        local_addr_t other_caculate_addr =
            other_local_addr[(stage_idx - 1) & 0x1];
        if (binary_type == BINARY_ADD || binary_type == BINARY_SUB) {
          // mul scalar
          const_binary_fp_func func_const =
              get_const_binary_fp_func(BINARY_MUL, false);
          func_const(buffer_local_addr, other_local_addr[(stage_idx - 1) & 0x1],
                     value, &cur_shape, NULL, NULL, dtype);
          other_caculate_addr = buffer_local_addr;
        }

        // binary op
        binary_fp_func func = get_binary_fp_func(binary_type);
        func(out_local_addr[(stage_idx - 1) & 0x1],
             in_local_addr[(stage_idx - 1) & 0x1], other_caculate_addr,
             &cur_shape, NULL, NULL, NULL, dtype);
      } else {
        // dtype int
        local_addr_t other_caculate_addr =
            other_local_addr[(stage_idx - 1) & 0x1];

        if (binary_type == BINARY_ADD || binary_type == BINARY_SUB) {
          // mul scalar
          const_binary_int_func func_const =
              get_const_binary_int_func(BINARY_MUL, false);
          func_const(buffer_local_addr, other_local_addr[(stage_idx - 1) & 0x1],
                     value, &cur_shape, NULL, NULL, dtype, dtype, dtype, 0,
                     NO_USE, false);
          other_caculate_addr = buffer_local_addr;
        }

        // for int sub op, the dst'type only support int8, int16, int32
        data_type_t dst_dtype = dtype;
        if (binary_type == BINARY_SUB) {
          if (dtype == DT_UINT8) {
            dst_dtype = DT_INT8;
          }
          if (dtype == DT_UINT16) {
            dst_dtype = DT_INT16;
          }
        }

        // binary op
        binary_int_func func = get_binary_int_func(binary_type);
        func(out_local_addr[(stage_idx - 1) & 0x1],
             in_local_addr[(stage_idx - 1) & 0x1], other_caculate_addr,
             &cur_shape, NULL, NULL, NULL, dst_dtype, dtype, dtype, 0, NO_USE,
             false);
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

#ifndef BACKEND_SG2260
int tpu_kernel_api_binary(const void *args) {
  sg_api_binary_t *api = (sg_api_binary_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_INT8 || api->dtype == DT_UINT8 ||
                   api->dtype == DT_INT16 || api->dtype == DT_UINT16 ||
                   api->dtype == DT_INT32 || api->dtype == DT_UINT32 ||
                   api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                   api->dtype == DT_FP32);
  scalar_t value;
  if (api->dtype == DT_FP32 || api->dtype == DT_BFP16 ||
      api->dtype == DT_FP16) {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32,
                        RM_HALF_TO_EVEN);
  } else {
    scalar_t value_s32 = {.s32 = api->value};
    value = tpu_int_cast(value_s32, (data_type_t)api->dtype, DT_INT32);
  }
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  if ((sg_binary_type_t)api->binary_type == BINARY_DIV) {
    nodechip_binary_div(api->input_global_addr, api->other_global_addr,
                        api->output_global_addr, length,
                        (data_type_t)api->dtype);
  } else {
    nodechip_binary(api->input_global_addr, api->other_global_addr,
                    api->output_global_addr, value, length,
                    (data_type_t)api->dtype,
                    (sg_binary_type_t)api->binary_type);
  }
  tpu_poll();
  return 0;
}

#else // defined BACKEND_SG2260

int tpu_kernel_api_binary_multi_core(const void *args) {
  sg_api_binary_t *api = (sg_api_binary_t *)args;
  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  TPUKERNEL_ASSERT(api->dtype == DT_INT8 || api->dtype == DT_UINT8 ||
                   api->dtype == DT_INT16 || api->dtype == DT_UINT16 ||
                   api->dtype == DT_INT32 || api->dtype == DT_UINT32 ||
                   api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                   api->dtype == DT_FP32);
  scalar_t value;
  if (api->dtype == DT_FP32 || api->dtype == DT_BFP16 ||
      api->dtype == DT_FP16) {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32,
                        RM_HALF_TO_EVEN);
  } else {
    scalar_t value_s32 = {.s32 = api->value};
    value = tpu_int_cast(value_s32, (data_type_t)api->dtype, DT_INT32);
  }
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  const int dsize = tpu_data_type_size((data_type_t)api->dtype);
  const int div_out_dsize = (api->dtype == DT_FP16 || api->dtype == DT_BFP16)
                                ? dsize
                                : tpu_data_type_size(DT_FP32);

  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();

  long long length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);

  long long cur_length_slice = length_slice;
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }

  if (core_idx < length_secs) {
    if ((sg_binary_type_t)api->binary_type == BINARY_DIV) {
      nodechip_binary_div(
          api->input_global_addr + (length_slice * core_idx) * dsize,
          api->other_global_addr + (length_slice * core_idx) * dsize,
          api->output_global_addr + (length_slice * core_idx) * div_out_dsize,
          cur_length_slice, (data_type_t)api->dtype);
    } else {
      nodechip_binary(
          api->input_global_addr + (length_slice * core_idx) * dsize,
          api->other_global_addr + (length_slice * core_idx) * dsize,
          api->output_global_addr + (length_slice * core_idx) * dsize, value,
          cur_length_slice, (data_type_t)api->dtype,
          (sg_binary_type_t)api->binary_type);
    }
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary_multi_core);

int tpu_kernel_api_binary(const void *args)
{
    return tpu_kernel_api_binary_multi_core(args);
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary);
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

void nodechip_binary_bcast_div(global_addr_t input_global_addr,
                               global_addr_t other_global_addr,
                               global_addr_t output_global_addr,
                               dim4 *input_shape, dim4 *other_shape,
                               dim4 *output_shape, dim4 *input_global_stride,
                               dim4 *other_global_stride,
                               dim4 *output_global_stride, data_type_t dtype) {
  bool input_bcast[4] = {
      input_shape->n != output_shape->n, input_shape->c != output_shape->c,
      input_shape->h != output_shape->h, input_shape->w != output_shape->w};
  bool other_bcast[4] = {
      other_shape->n != output_shape->n, other_shape->c != output_shape->c,
      other_shape->h != output_shape->h, other_shape->w != output_shape->w};

  int input_size = 0, output_size = 0, buffer_size = 0;
  int buffer_num = 2;
  data_type_t save_dtype = DT_FP32;
  if (dtype == DT_FP32) {
    // 4*input + 2*output + 0*buffer = 12 bank
    input_size = 2 * BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 0;
    buffer_num = 0;
  } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
    // 4*input + 2*output + 3*buffer = 12 bank
    input_size = BANK_SIZE;
    output_size = BANK_SIZE;
    buffer_size = 2 * BANK_SIZE;
    buffer_num = 3;
    save_dtype = dtype;
  } else if (dtype == DT_INT32) {
    // 4*input + 2*output + 2*buffer = 16 bank
    input_size = 2 * BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 2 * BANK_SIZE;
  } else if (dtype == DT_INT16 || dtype == DT_UINT16) {
    // 4*input + 2*output + 2*buffer = 12 bank
    input_size = BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 2 * BANK_SIZE;
  } else if (dtype == DT_INT8 || dtype == DT_UINT8) {
    // 4*input + 2*output + 2*buffer = 16 bank
    input_size = BANK_SIZE;
    output_size = 3 * BANK_SIZE;
    buffer_size = 3 * BANK_SIZE;
  }

  TPUKERNEL_ASSERT(input_size * 4 + output_size * 2 +
                       buffer_size * buffer_num <=
                   LOCAL_MEM_SIZE);

  int dtype_size = tpu_data_type_size(dtype);
  int load_size = (dtype_size == 1) ? input_size * 3 / 4 : input_size;

  local_addr_t input_local_addr[2] = {0, 1 * input_size};
  local_addr_t other_local_addr[2] = {2 * input_size, 3 * input_size};
  local_addr_t output_local_addr[2] = {4 * input_size,
                                       4 * input_size + output_size};
  local_addr_t buffer_start_addr = output_local_addr[1] + output_size;
  local_addr_t buffer_fp32_local_addr[3] = {
      buffer_start_addr, buffer_start_addr + buffer_size,
      buffer_start_addr + 2 * buffer_size};

  const int c_per_npu = DIV_UP(output_shape->c, NPU_NUM);
  const int eu_num = tpu_eu_num(DT_FP32);
  int nmax = output_shape->n, hmax = output_shape->h,
      cmax = c_per_npu * NPU_NUM,
      wmax = MIN(output_shape->w, tpu_gdma_shape_limit(TENSOR_W_DIM));

  while (true) {
    int size = tpu_aligned_feature_size(hmax, wmax, dtype) *
               DIV_UP(cmax, NPU_NUM) * nmax;
    if (size <= load_size) {
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
  dim4 gdma_shape, slice_shape;
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
          tpu_aligned_stride(&input_local_stride, 0, &slice_shape, dtype);
          input_local_shape.n = input_bcast[0] ? 1 : slice_shape.n;
          input_local_shape.c = input_bcast[1] ? 1 : slice_shape.c;
          input_local_shape.h = input_bcast[2] ? 1 : slice_shape.h;
          input_local_shape.w = input_bcast[3] ? 1 : slice_shape.w;
          global_addr_t input_global_addr_gdma =
              input_global_addr + ((input_bcast[0] ? 0 : ndone) *
                                       (long long)input_global_stride->n +
                                   (input_bcast[1] ? 0 : cdone) *
                                       (long long)input_global_stride->c +
                                   (input_bcast[2] ? 0 : hdone) *
                                       (long long)input_global_stride->h +
                                   (input_bcast[3] ? 0 : wdone) *
                                       (long long)input_global_stride->w) *
                                      tpu_data_type_size(dtype);
          tpu_gdma_cpy_S2L(input_local_addr[index], input_global_addr_gdma,
                           &input_local_shape, &input_local_stride,
                           input_global_stride, dtype);
          // Move other from global memory to local memory
          tpu_aligned_stride(&other_local_stride, 0, &slice_shape, dtype);
          other_local_shape.n = other_bcast[0] ? 1 : slice_shape.n;
          other_local_shape.c = other_bcast[1] ? 1 : slice_shape.c;
          other_local_shape.h = other_bcast[2] ? 1 : slice_shape.h;
          other_local_shape.w = other_bcast[3] ? 1 : slice_shape.w;
          global_addr_t other_global_addr_gdma =
              other_global_addr + ((other_bcast[0] ? 0 : ndone) *
                                       (long long)other_global_stride->n +
                                   (other_bcast[1] ? 0 : cdone) *
                                       (long long)other_global_stride->c +
                                   (other_bcast[2] ? 0 : hdone) *
                                       (long long)other_global_stride->h +
                                   (other_bcast[3] ? 0 : wdone) *
                                       (long long)other_global_stride->w) *
                                      tpu_data_type_size(dtype);
          tpu_gdma_cpy_S2L(other_local_addr[index], other_global_addr_gdma,
                           &other_local_shape, &other_local_stride,
                           other_global_stride, dtype);

          // Broadcast input if needed
          if (input_bcast[1]) {
            input_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast(input_local_addr[index], input_local_addr[index],
                              &input_local_shape, dtype);
          }
          if (input_bcast[0] || input_bcast[2] || input_bcast[3] ||
              (input_bcast[1] && slice_shape.c > NPU_NUM)) {
            for (int i = 0; i < 4; ++i) {
              ((int *)&input_local_stride)[i] =
                  input_bcast[i] ? 0 : ((int *)&input_local_stride)[i];
            }
          }

          // Broadcast other if needed
          if (other_bcast[1]) {
            other_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast(other_local_addr[index], other_local_addr[index],
                              &other_local_shape, dtype);
          }
          if (other_bcast[0] || other_bcast[2] || other_bcast[3] ||
              (other_bcast[1] && slice_shape.c > NPU_NUM)) {
            for (int i = 0; i < 4; ++i) {
              ((int *)&other_local_stride)[i] =
                  other_bcast[i] ? 0 : ((int *)&other_local_stride)[i];
            }
          }

          tpu_parallel_start();

          if (l2s) {
            // Move out from local memory to global memory
            tpu_gdma_cpy_L2S(output_global_addr_gdma,
                             output_local_addr[1 - index], &gdma_shape,
                             output_global_stride, NULL, save_dtype);
          }

          // #ifdef BACKEND_1684X
          if (dtype == DT_FP32) {
            tpu_bdc_fp32_div(output_local_addr[index], input_local_addr[index],
                             other_local_addr[index], &slice_shape, NULL,
                             &input_local_stride, &other_local_stride);
          } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
            tpu_bdc_cast(buffer_fp32_local_addr[0], input_local_addr[index],
                         &slice_shape, NULL, &input_local_stride, DT_FP32,
                         dtype, RM_HALF_TO_EVEN);
            tpu_bdc_cast(buffer_fp32_local_addr[1], other_local_addr[index],
                         &slice_shape, NULL, &other_local_stride, DT_FP32,
                         dtype, RM_HALF_TO_EVEN);
            tpu_bdc_fp32_div(
                buffer_fp32_local_addr[2], buffer_fp32_local_addr[0],
                buffer_fp32_local_addr[1], &slice_shape, NULL, NULL, NULL);
            tpu_bdc_cast(output_local_addr[index], buffer_fp32_local_addr[2],
                         &slice_shape, NULL, NULL, dtype, DT_FP32,
                         RM_HALF_TO_EVEN);
          } else {
            tpu_bdc_cast(buffer_fp32_local_addr[0], input_local_addr[index],
                         &slice_shape, NULL, &input_local_stride, DT_FP32,
                         dtype, RM_HALF_TO_EVEN);
            tpu_bdc_cast(buffer_fp32_local_addr[1], other_local_addr[index],
                         &slice_shape, NULL, &other_local_stride, DT_FP32,
                         dtype, RM_HALF_TO_EVEN);
            tpu_bdc_fp32_div(
                output_local_addr[index], buffer_fp32_local_addr[0],
                buffer_fp32_local_addr[1], &slice_shape, NULL, NULL, NULL);
          }

          tpu_parallel_end();

          output_global_addr_gdma =
              output_global_addr +
              (ndone * (long long)output_global_stride->n +
               cdone * (long long)output_global_stride->c +
               hdone * (long long)output_global_stride->h +
               wdone * (long long)output_global_stride->w) *
                  tpu_data_type_size(save_dtype);
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
                     &gdma_shape, output_global_stride, NULL, save_dtype);
  }
}

void nodechip_binary_bcast(global_addr_t input_global_addr,
                           global_addr_t other_global_addr,
                           global_addr_t output_global_addr, dim4 *input_shape,
                           dim4 *other_shape, dim4 *output_shape,
                           dim4 *input_global_stride, dim4 *other_global_stride,
                           dim4 *output_global_stride, scalar_t value,
                           data_type_t dtype, sg_binary_type_t binary_type) {
  bool input_bcast[4] = {
      input_shape->n != output_shape->n, input_shape->c != output_shape->c,
      input_shape->h != output_shape->h, input_shape->w != output_shape->w};
  bool other_bcast[4] = {
      other_shape->n != output_shape->n, other_shape->c != output_shape->c,
      other_shape->h != output_shape->h, other_shape->w != output_shape->w};

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int tensor_num = 2 + 2 + 2;
  if (binary_type == BINARY_ADD || binary_type == BINARY_SUB) {
    tensor_num += 1;
  }

  int tensor_bsize_pnpu = tpu_bank_num() / tensor_num * bank_bsize;
  TPUKERNEL_ASSERT(tensor_bsize_pnpu > 0);

  local_addr_t input_local_addr[2] = {0, 1 * tensor_bsize_pnpu};
  local_addr_t other_local_addr[2] = {2 * tensor_bsize_pnpu,
                                      3 * tensor_bsize_pnpu};
  local_addr_t output_local_addr[2] = {4 * tensor_bsize_pnpu,
                                       5 * tensor_bsize_pnpu};

  local_addr_t buffer_local_addr = 6 * tensor_bsize_pnpu;

  const int c_per_npu = DIV_UP(output_shape->c, NPU_NUM);
  const int eu_num = tpu_eu_num(dtype);
  int nmax = output_shape->n, hmax = output_shape->h,
      cmax = c_per_npu * NPU_NUM,
      wmax = MIN(output_shape->w, tpu_gdma_shape_limit(TENSOR_W_DIM));

  while (true) {
    int size = tpu_aligned_feature_size(hmax, wmax, dtype) *
               DIV_UP(cmax, NPU_NUM) * nmax;
    if (size <= tensor_bsize_pnpu) {
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
  dim4 gdma_shape, slice_shape;
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
          tpu_aligned_stride(&input_local_stride, 0, &slice_shape, dtype);
          input_local_shape.n = input_bcast[0] ? 1 : slice_shape.n;
          input_local_shape.c = input_bcast[1] ? 1 : slice_shape.c;
          input_local_shape.h = input_bcast[2] ? 1 : slice_shape.h;
          input_local_shape.w = input_bcast[3] ? 1 : slice_shape.w;
          global_addr_t input_global_addr_gdma =
              input_global_addr + ((input_bcast[0] ? 0 : ndone) *
                                       (long long)input_global_stride->n +
                                   (input_bcast[1] ? 0 : cdone) *
                                       (long long)input_global_stride->c +
                                   (input_bcast[2] ? 0 : hdone) *
                                       (long long)input_global_stride->h +
                                   (input_bcast[3] ? 0 : wdone) *
                                       (long long)input_global_stride->w) *
                                      tpu_data_type_size(dtype);
          tpu_gdma_cpy_S2L(input_local_addr[index], input_global_addr_gdma,
                           &input_local_shape, &input_local_stride,
                           input_global_stride, dtype);
          // Move other from global memory to local memory
          tpu_aligned_stride(&other_local_stride, 0, &slice_shape, dtype);
          other_local_shape.n = other_bcast[0] ? 1 : slice_shape.n;
          other_local_shape.c = other_bcast[1] ? 1 : slice_shape.c;
          other_local_shape.h = other_bcast[2] ? 1 : slice_shape.h;
          other_local_shape.w = other_bcast[3] ? 1 : slice_shape.w;
          global_addr_t other_global_addr_gdma =
              other_global_addr + ((other_bcast[0] ? 0 : ndone) *
                                       (long long)other_global_stride->n +
                                   (other_bcast[1] ? 0 : cdone) *
                                       (long long)other_global_stride->c +
                                   (other_bcast[2] ? 0 : hdone) *
                                       (long long)other_global_stride->h +
                                   (other_bcast[3] ? 0 : wdone) *
                                       (long long)other_global_stride->w) *
                                      tpu_data_type_size(dtype);
          tpu_gdma_cpy_S2L(other_local_addr[index], other_global_addr_gdma,
                           &other_local_shape, &other_local_stride,
                           other_global_stride, dtype);

          // Broadcast input if needed
          if (input_bcast[1]) {
            input_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast(input_local_addr[index], input_local_addr[index],
                              &input_local_shape, dtype);
          }
          if (input_bcast[0] || input_bcast[2] || input_bcast[3] ||
              (input_bcast[1] && slice_shape.c > NPU_NUM)) {
            for (int i = 0; i < 4; ++i) {
              ((int *)&input_local_stride)[i] =
                  input_bcast[i] ? 0 : ((int *)&input_local_stride)[i];
            }
          }

          // Broadcast other if needed
          if (other_bcast[1]) {
            other_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast(other_local_addr[index], other_local_addr[index],
                              &other_local_shape, dtype);
          }
          if (other_bcast[0] || other_bcast[2] || other_bcast[3] ||
              (other_bcast[1] && slice_shape.c > NPU_NUM)) {
            for (int i = 0; i < 4; ++i) {
              ((int *)&other_local_stride)[i] =
                  other_bcast[i] ? 0 : ((int *)&other_local_stride)[i];
            }
          }

          tpu_parallel_start();

          if (l2s) {
            // Move out from local memory to global memory
            tpu_gdma_cpy_L2S(output_global_addr_gdma,
                             output_local_addr[1 - index], &gdma_shape,
                             output_global_stride, NULL, dtype);
          }

          if (dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16) {
            local_addr_t other_calculate_addr = other_local_addr[index];
            if (binary_type == BINARY_ADD || binary_type == BINARY_SUB) {
              // mul scalar
              const_binary_fp_func func_const =
                  get_const_binary_fp_func(BINARY_MUL, false);
              func_const(buffer_local_addr, other_local_addr[index], value,
                         &slice_shape, NULL, NULL, dtype);
              other_calculate_addr = buffer_local_addr;
            }

            // binary op
            binary_fp_func func = get_binary_fp_func(binary_type);
            func(output_local_addr[index], input_local_addr[index],
                 other_calculate_addr, &slice_shape, NULL, &input_local_stride,
                 &other_local_stride, dtype);

          } else {

            local_addr_t other_calculate_addr = other_local_addr[index];
            if (binary_type == BINARY_ADD || binary_type == BINARY_SUB) {
              // mul scalar
              const_binary_int_func func_const =
                  get_const_binary_int_func(BINARY_MUL, false);
              func_const(buffer_local_addr, other_local_addr[index], value,
                         &slice_shape, NULL, NULL, dtype, dtype,
                         dtype, 0, NO_USE, false);
              other_calculate_addr = buffer_local_addr;
            }

            // for int sub op, the dst'type only support int8, int16, int32
            data_type_t dst_dtype = dtype;
            if (binary_type == BINARY_SUB) {
              if (dtype == DT_UINT8) {
                dst_dtype = DT_INT8;
              }
              if (dtype == DT_UINT16) {
                dst_dtype = DT_INT16;
              }
            }

            // binary op
            binary_int_func func = get_binary_int_func(binary_type);
            func(output_local_addr[index], input_local_addr[index],
                 other_calculate_addr, &slice_shape, NULL, &input_local_stride,
                 &other_local_stride, dst_dtype, dtype, dtype, 0, NO_USE,
                 false);
          }
          tpu_parallel_end();

          output_global_addr_gdma =
              output_global_addr +
              (ndone * (long long)output_global_stride->n +
               cdone * (long long)output_global_stride->c +
               hdone * (long long)output_global_stride->h +
               wdone * (long long)output_global_stride->w) *
                  tpu_data_type_size(dtype);
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
                     &gdma_shape, output_global_stride, NULL, dtype);
  }
}

int tpu_kernel_api_binary_bcast(const void *args) {
  sg_api_binary_bcast_t *api = (sg_api_binary_bcast_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_INT8 || api->dtype == DT_UINT8 ||
                   api->dtype == DT_INT16 || api->dtype == DT_UINT16 ||
                   api->dtype == DT_INT32 || api->dtype == DT_UINT32 ||
                   api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                   api->dtype == DT_FP32);
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

  scalar_t value;
  if (api->dtype == DT_FP32 || api->dtype == DT_BFP16 ||
      api->dtype == DT_FP16) {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32,
                        RM_HALF_TO_EVEN);
  } else {
    scalar_t value_s32 = {.s32 = api->value};
    value = tpu_int_cast(value_s32, (data_type_t)api->dtype, DT_INT32);
  }

  dim4 input_global_stride, other_global_stride, output_global_stride;
  tpu_continuous_stride(&input_global_stride, &input_shape);
  tpu_continuous_stride(&other_global_stride, &other_shape);
  tpu_continuous_stride(&output_global_stride, &output_shape);

  tpu_initialize();
  if ((sg_binary_type_t)api->binary_type == BINARY_DIV) {
    nodechip_binary_bcast_div(
        api->input_global_addr, api->other_global_addr, api->output_global_addr,
        &input_shape, &other_shape, &output_shape, &input_global_stride,
        &other_global_stride, &output_global_stride, (data_type_t)api->dtype);
  } else {
    nodechip_binary_bcast(
        api->input_global_addr, api->other_global_addr, api->output_global_addr,
        &input_shape, &other_shape, &output_shape, &input_global_stride,
        &other_global_stride, &output_global_stride, value,
        (data_type_t)api->dtype, (sg_binary_type_t)api->binary_type);
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary_bcast);

#ifdef BACKEND_SG2260
int tpu_kernel_api_binary_bcast_multi_core(const void *args) {
  sg_api_binary_bcast_t *api = (sg_api_binary_bcast_t *)args;
  tpu_initialize();
  TPUKERNEL_ASSERT(api->dtype == DT_INT8 || api->dtype == DT_UINT8 ||
                   api->dtype == DT_INT16 || api->dtype == DT_UINT16 ||
                   api->dtype == DT_INT32 || api->dtype == DT_UINT32 ||
                   api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                   api->dtype == DT_FP32);
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

  scalar_t value;
  if (api->dtype == DT_FP32 || api->dtype == DT_BFP16 ||
      api->dtype == DT_FP16) {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32,
                        RM_HALF_TO_EVEN);
  } else {
    scalar_t value_s32 = {.s32 = api->value};
    value = tpu_int_cast(value_s32, (data_type_t)api->dtype, DT_INT32);
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

  unsigned long long length = input_is_one ? ((int *)&other_shape)[split_dim]
                            : ((int *)&input_shape)[split_dim];
  long long length_slice = DIV_UP(length, core_num);
  int allocate_core = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(allocate_core <= core_num);

  int last_length_slice = length - length_slice * (allocate_core - 1);

  dim4 input_stride = {0}, other_stride = {0}, output_stride = {0};
  tpu_continuous_stride(&input_stride, &input_shape);
  tpu_continuous_stride(&other_stride, &other_shape);
  tpu_continuous_stride(&output_stride, &output_shape);

  // mutil core
  const int dsize = tpu_data_type_size((data_type_t)api->dtype);
  const int div_out_dsize = (api->dtype == DT_FP16 || api->dtype == DT_BFP16)
                                ? dsize
                                : tpu_data_type_size(DT_FP32);
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
      input_is_one ? 0 : ((int *)&input_stride)[split_dim] * dsize;
  size_t other_offset =
      other_is_one ? 0 : ((int *)&other_stride)[split_dim] * dsize;
  size_t output_offset = ((int *)&output_stride)[split_dim] * dsize;
  size_t div_output_offset = ((int *)&output_stride)[split_dim] * div_out_dsize;

  if (core_idx < allocate_core) {
    if ((sg_binary_type_t)api->binary_type == BINARY_DIV) {
      nodechip_binary_bcast_div(
          api->input_global_addr + core_idx * length_slice * input_offset,
          api->other_global_addr + core_idx * length_slice * other_offset,
          api->output_global_addr + core_idx * length_slice * div_output_offset,
          &input_shape, &other_shape, &output_shape, &input_stride,
          &other_stride, &output_stride, (data_type_t)api->dtype);

    } else {
      nodechip_binary_bcast(
          api->input_global_addr + core_idx * length_slice * input_offset,
          api->other_global_addr + core_idx * length_slice * other_offset,
          api->output_global_addr + core_idx * length_slice * output_offset,
          &input_shape, &other_shape, &output_shape, &input_stride,
          &other_stride, &output_stride, value, (data_type_t)api->dtype,
          (sg_binary_type_t)api->binary_type);
    }
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary_bcast_multi_core);
#endif

void nodechip_binary_scalar_div(global_addr_t input_global_addr,
                                global_addr_t output_global_addr, float value,
                                unsigned long long length, data_type_t dtype,
                                bool inversed) {
  if (length == 0) {
    return;
  }

  int input_size = 0, output_size = 0, buffer_size = 0;
  int buffer_num = 1;
  data_type_t save_dtype = DT_FP32;
  if (dtype == DT_FP32) {
    // 2*input + 2*output = 16 bank
    input_size = 4 * BANK_SIZE;
    output_size = 4 * BANK_SIZE;
    buffer_size = 0;
    buffer_num = 0;
  } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
    // 2*input + 2*output + 2*buffer = 16 bank
    input_size = 2 * BANK_SIZE;
    output_size = 2 * BANK_SIZE;
    buffer_size = 4 * BANK_SIZE;
    buffer_num = 2;
    save_dtype = dtype;
  } else if (dtype == DT_INT32) {
    // 2*input + 2*output + buffer = 15 bank
    input_size = 3 * BANK_SIZE;
    output_size = 3 * BANK_SIZE;
    buffer_size = 3 * BANK_SIZE;
  } else if (dtype == DT_INT16 || dtype == DT_UINT16) {
    // 2*input + 2*output + buffer = 16 bank
    input_size = 2 * BANK_SIZE;
    output_size = 4 * BANK_SIZE;
    buffer_size = 4 * BANK_SIZE;
  } else if (dtype == DT_INT8 || dtype == DT_UINT8) {
    // 2*input + 2*output + buffer = 14 bank
    input_size = BANK_SIZE;
    output_size = 4 * BANK_SIZE;
    buffer_size = 4 * BANK_SIZE;
  } else {
    TPUKERNEL_ASSERT(false);
  }

  TPUKERNEL_ASSERT(input_size * 2 + output_size * 2 +
                       buffer_size * buffer_num <=
                   LOCAL_MEM_SIZE);

  int dtype_size = tpu_data_type_size(dtype);

  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;
  const unsigned int w_dim = tpu_eu_num(dtype);
  const int n_dim = input_size * tpu_npu_num() / tpu_data_type_size(DT_FP32) / max_m_dim;

  local_addr_t in_local_addr[2] = {0, 1 * input_size};
  local_addr_t out_local_addr[2] = {2 * input_size,
                                    2 * input_size + output_size};
  local_addr_t buffer_start_addr = out_local_addr[1] + output_size;
  local_addr_t buffer_fp32_local_addr[2] = {buffer_start_addr,
                                            buffer_start_addr + buffer_size};

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
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0]; // cur_len / cur_m_dim[0] >= 1
    }

    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(
          output_global_addr + cur_idx[2] * tpu_data_type_size(save_dtype),
          out_local_addr[stage_idx & 0x1],
          /**rows, cols, cols_per_channel, row_stride*/
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2], save_dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(in_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // calculate
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)

      const_binary_fp32_div_func func_const =
          get_const_binary_fp32_div_func(inversed);

      if (dtype == DT_FP32) {
        func_const(out_local_addr[(stage_idx - 1) & 0x1],
                   in_local_addr[(stage_idx - 1) & 0x1], value, &cur_shape,
                   NULL, NULL);
      } else if (dtype == DT_FP16 || dtype == DT_BFP16) {
        tpu_bdc_cast(buffer_fp32_local_addr[0],
                     in_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        func_const(buffer_fp32_local_addr[1], buffer_fp32_local_addr[0], value,
                   &cur_shape, NULL, NULL);
        tpu_bdc_cast(out_local_addr[(stage_idx - 1) & 0x1],
                     buffer_fp32_local_addr[1], &cur_shape, NULL, NULL, dtype,
                     DT_FP32, RM_HALF_TO_EVEN);
      } else {
        tpu_bdc_cast(buffer_fp32_local_addr[0],
                     in_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        func_const(out_local_addr[(stage_idx - 1) & 0x1],
                   buffer_fp32_local_addr[0], value, &cur_shape, NULL, NULL);
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

void nodechip_binary_scalar(global_addr_t input_global_addr,
                            global_addr_t output_global_addr, scalar_t value,
                            unsigned long long length, data_type_t dtype,
                            sg_binary_type_t binary_type, bool inversed) {
  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_num = 2 + 2; // 2 inputs, 2 outputs
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
      // case n = 1 NOTE: n_dim * max_m_dim <= tensor_size_pnpu * npu_num,
      // it's always a legal size.
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0]; // cur_len / cur_m_dim[0] >= 1
    }

    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(output_global_addr + cur_idx[2] * dtype_size,
                          out_local_addr[stage_idx & 0x1],
                          /**rows, cols, cols_per_channel, row_stride*/
                          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],
                          dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(in_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)

      if (dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16) {
        const_binary_fp_func func_const =
            get_const_binary_fp_func(binary_type, inversed);
        func_const(out_local_addr[(stage_idx - 1) & 0x1],
                   in_local_addr[(stage_idx - 1) & 0x1], value, &cur_shape,
                   NULL, NULL, dtype);

      } else {
        // for int sub op, the dst'type only support int8, int16, int32
        data_type_t dst_dtype = dtype;
        if (binary_type == BINARY_SUB) {
          if (dtype == DT_UINT8) {
            dst_dtype = DT_INT8;
          }
          if (dtype == DT_UINT16) {
            dst_dtype = DT_INT16;
          }
        }

        const_binary_int_func func_const =
            get_const_binary_int_func(binary_type, inversed);
        func_const(out_local_addr[(stage_idx - 1) & 0x1],
                   in_local_addr[(stage_idx - 1) & 0x1], value, &cur_shape,
                   NULL, NULL, dst_dtype, dtype, dtype, 0, NO_USE, false);
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

int tpu_kernel_api_binary_c(const void *args) {
  sg_api_binary_c_t *api = (sg_api_binary_c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_INT8 || api->dtype == DT_UINT8 ||
                   api->dtype == DT_INT16 || api->dtype == DT_UINT16 ||
                   api->dtype == DT_INT32 || api->dtype == DT_UINT32 ||
                   api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                   api->dtype == DT_FP32);
  scalar_t value;
  if (api->dtype == DT_FP32 || api->dtype == DT_BFP16 ||
      api->dtype == DT_FP16) {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32,
                        RM_HALF_TO_EVEN);
  } else {
    scalar_t value_s32 = {.s32 = api->value};
    value = tpu_int_cast(value_s32, (data_type_t)api->dtype, DT_INT32);
  }

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  if ((sg_binary_type_t)api->binary_type == BINARY_DIV) {
    nodechip_binary_scalar_div(api->input_global_addr, api->output_global_addr,
                               api->value, length, (data_type_t)api->dtype,
                               api->inversed);
  } else {
    nodechip_binary_scalar(api->input_global_addr, api->output_global_addr,
                           value, length, (data_type_t)api->dtype,
                           (sg_binary_type_t)api->binary_type, api->inversed);
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary_c);

#ifdef BACKEND_SG2260
int tpu_kernel_api_binary_c_multi_core(const void *args) {
  sg_api_binary_c_t *api = (sg_api_binary_c_t *)args;
  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  TPUKERNEL_ASSERT(api->dtype == DT_INT8 || api->dtype == DT_UINT8 ||
                   api->dtype == DT_INT16 || api->dtype == DT_UINT16 ||
                   api->dtype == DT_INT32 || api->dtype == DT_UINT32 ||
                   api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                   api->dtype == DT_FP32);
  scalar_t value;
  if (api->dtype == DT_FP32 || api->dtype == DT_BFP16 ||
      api->dtype == DT_FP16) {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32,
                        RM_HALF_TO_EVEN);
  } else {
    scalar_t value_s32 = {.s32 = api->value};
    value = tpu_int_cast(value_s32, (data_type_t)api->dtype, DT_INT32);
  }
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  const int dsize = tpu_data_type_size((data_type_t)api->dtype);
  const int out_div_dsize = (api->dtype == DT_FP16 || api->dtype == DT_BFP16)
                                ? dsize
                                : tpu_data_type_size(DT_FP32);
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();

  long long length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);

  long long cur_length_slice = length_slice;
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }

  if (core_idx < length_secs) {
    if ((sg_binary_type_t)api->binary_type == BINARY_DIV) {
      nodechip_binary_scalar_div(
          api->input_global_addr + (length_slice * core_idx) * dsize,
          api->output_global_addr + (length_slice * core_idx) * out_div_dsize,
          api->value, cur_length_slice, (data_type_t)api->dtype, api->inversed);
    } else {
      nodechip_binary_scalar(
          api->input_global_addr + (length_slice * core_idx) * dsize,
          api->output_global_addr + (length_slice * core_idx) * dsize, value,
          cur_length_slice, (data_type_t)api->dtype,
          (sg_binary_type_t)api->binary_type, api->inversed);
    }
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary_c_multi_core);
#endif
