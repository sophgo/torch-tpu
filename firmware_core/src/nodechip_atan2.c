#include "sg_api_struct.h"
#include "tpu_kernel.h"


#define PI 3.14159265

void nodechip_atan2c(scalar_t value, global_addr_t other_global_addr,
                     global_addr_t output_global_addr, int length,
                     data_type_t output_dtype) {
  if (length == 0) return;
  int npu_num = tpu_npu_num();
  int bank_num = tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu() / bank_num;
  int tensor_num = 2 + 2 + 4;  // 2 inputs, 2 outputs, 4 buffer
  int coeff_bank_num = 4;      // 4 coeff
  int tensor_size = (bank_num - coeff_bank_num) / tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size > 0);

  local_addr_t other_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t work_local_addrs[4] = {4 * tensor_size, 5 * tensor_size,
                                      6 * tensor_size, 7 * tensor_size};
  local_addr_t exp_coeff_local_addr = 8 * tensor_size;
  local_addr_t log_coeff_local_addr = 8 * tensor_size + 1 * bank_size;
  local_addr_t exp_table_local_addr = 8 * tensor_size + 2 * bank_size;
  local_addr_t arcsin_coeff_local_addr = 8 * tensor_size + 3 * bank_size;

  data_type_t dtype = DT_FP32;
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_w =
      DIV_UP(MIN(length, tensor_size * npu_num / dtype_size), npu_num);

  float scalar = value.f32;
  int todo = length;
  int done = 0;
  dim4 shape = {.n = 1, .h = 1};
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  float POW_C = 2;
  scalar_t ADD_C = {.f32 = 1.f};
  scalar_t positive_zero = {.u32 = 0x00000000};
  scalar_t negtive_zero = {.u32 = 0x80000000};
  scalar_t true_val = {.u8 = 1};
  scalar_t PI_C = {.f32 = (scalar > 0 || value.u32 == 0x00000000) ? PI : -PI};
  scalar_t half_PI_C = {
      .f32 = (scalar > 0 || value.u32 == 0x00000000) ? PI / 2 : -PI / 2};
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
  tpu_bdc_load_fp32_arcsin_coeff(arcsin_coeff_local_addr);
  while (todo != 0) {
    if (todo > NPU_NUM) {
      shape.c = NPU_NUM;
      shape.w = MIN(todo / NPU_NUM, tensor_w);
    } else {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L(other_local_addrs[index],
                     other_global_addr + done * dtype_size, &shape, NULL, NULL,
                     dtype);
    if (tpu_is_parallel_state()) {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if (l2s) {
      tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL,
                       output_dtype);
    }

    // y/x
    tpu_bdc_fp32_C_div(work_local_addrs[3], other_local_addrs[index], scalar,
                       &shape, NULL, NULL);

    // atan
    tpu_bdc_abs(work_local_addrs[2], work_local_addrs[3], &shape, NULL, NULL,
                dtype);
    tpu_bdc_fp32_pow_C(output_local_addrs[index], work_local_addrs[2],
                       work_local_addrs[0], work_local_addrs[1],
                       exp_coeff_local_addr, log_coeff_local_addr,
                       exp_table_local_addr, POW_C, &shape);
    tpu_bdc_fp_add_C(work_local_addrs[2], output_local_addrs[index], ADD_C,
                     &shape, NULL, NULL, dtype);
    tpu_bdc_fp32_rsqrt(output_local_addrs[index], work_local_addrs[2], &shape);
    tpu_bdc_fp_mul(work_local_addrs[2], output_local_addrs[index],
                   work_local_addrs[3], &shape, NULL, NULL, NULL, dtype);
    tpu_bdc_fp32_arcsin(output_local_addrs[index], work_local_addrs[2],
                        work_local_addrs[1], arcsin_coeff_local_addr, &shape);

    // y/x==-0 atan(y/x)==-0
    tpu_bdc_equal_C(work_local_addrs[0], work_local_addrs[3], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_set_C(work_local_addrs[1], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[2], negtive_zero, &shape, NULL, DT_UINT32);
    variable_t src0 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[0]}};
    variable_t src1 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[1]}};
    variable_t src2 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[2]}};
    variable_t src3 = {.type = TENSOR,
                       .context = {.addr = output_local_addrs[index]}};
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x < 0
    tpu_bdc_set_C(work_local_addrs[0], positive_zero, &shape, NULL, dtype);
    tpu_bdc_fp_add_C(work_local_addrs[1], output_local_addrs[0], PI_C, &shape,
                     NULL, NULL, dtype);
    src0.context.addr = other_local_addrs[index];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_less_select(work_local_addrs[2], &src0, &src1, &src2, &src3, &shape,
                        dtype, dtype);

    if (scalar == 0) {
      if (value.u32 >> 31 == 0) {
        // x == +0 y == +0
        tpu_bdc_equal_C(output_local_addrs[index], other_local_addrs[index],
                        positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                        dtype);
        tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work_local_addrs[1], positive_zero, &shape, NULL,
                      DT_UINT32);
        src0.context.addr = output_local_addrs[index];
        src1.context.addr = work_local_addrs[0];
        src2.context.addr = work_local_addrs[1];
        src3.context.addr = work_local_addrs[2];
        tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                             &shape, DT_UINT8, dtype);
      } else {
        // x == +0 y == -0
        tpu_bdc_equal_C(output_local_addrs[index], other_local_addrs[index],
                        positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                        dtype);
        tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work_local_addrs[1], negtive_zero, &shape, NULL,
                      DT_UINT32);
        src0.context.addr = output_local_addrs[index];
        src1.context.addr = work_local_addrs[0];
        src2.context.addr = work_local_addrs[1];
        src3.context.addr = work_local_addrs[2];
        tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                             &shape, DT_UINT8, dtype);
      }
      // x == -0   y == +0 || y == -0
      tpu_bdc_equal_C(work_local_addrs[1], other_local_addrs[index],
                      negtive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                      DT_UINT32);
      tpu_bdc_set_C(work_local_addrs[2], PI_C, &shape, NULL, dtype);
      src0.context.addr = work_local_addrs[1];
      src2.context.addr = work_local_addrs[2];
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2,
                           &src3, &shape, DT_UINT8, dtype);
    } else {
      // x == 0 y > 0
      tpu_bdc_set_C(work_local_addrs[0], positive_zero, &shape, NULL, dtype);
      tpu_bdc_set_C(work_local_addrs[1], half_PI_C, &shape, NULL, dtype);
      src0.context.addr = other_local_addrs[index];
      src1.context.addr = work_local_addrs[0];
      src2.context.addr = work_local_addrs[1];
      src3.context.addr = work_local_addrs[2];
      tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2,
                           &src3, &shape, dtype, dtype);
    }

    if (output_dtype != dtype) {
      tpu_bdc_cast(output_local_addrs[index], output_local_addrs[index], &shape,
                   NULL, NULL, output_dtype, dtype, RM_HALF_TO_EVEN);
    }

    l2s = true;
    l2s_global_addr =
        output_global_addr + done * tpu_data_type_size(output_dtype);
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }
  if (l2s) {
    tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL,
                     output_dtype);
  }
}

int tpu_kernel_api_atan2c(const void *args) {
  sg_api_atan2c_t *api = (sg_api_atan2c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  scalar_t value = {.f32 = api->scalar};
  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  nodechip_atan2c(value, api->other_global_addr, api->output_global_addr,
                  length, (data_type_t)api->dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2c);

#ifdef BACKEND_SG2260
int tpu_kernel_api_atan2c_multi_core(const void *args) {
  sg_api_atan2c_t *api = (sg_api_atan2c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  scalar_t value = {.f32 = api->scalar};
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
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }
  if (core_idx * length_slice < length) {
    nodechip_atan2c(
        value,
        api->other_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(DT_FP32),
        api->output_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        cur_length_slice, (data_type_t)api->dtype);
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2c_multi_core);
#endif

void nodechip_atan2_c(global_addr_t input_global_addr, scalar_t value,
                      global_addr_t output_global_addr, int length,
                      data_type_t output_dtype) {
  if (length == 0) return;
  int npu_num = tpu_npu_num();
  int bank_num = tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu() / bank_num;
  int tensor_num = 2 + 2 + 4;  // 2 inputs, 2 outputs, 4 buffer
  int coeff_bank_num = 4;      // 4 coeff
  int tensor_size = (bank_num - coeff_bank_num) / tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size > 0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t work_local_addrs[4] = {4 * tensor_size, 5 * tensor_size,
                                      6 * tensor_size, 7 * tensor_size};
  local_addr_t exp_coeff_local_addr = 8 * tensor_size;
  local_addr_t log_coeff_local_addr = 8 * tensor_size + 1 * bank_size;
  local_addr_t exp_table_local_addr = 8 * tensor_size + 2 * bank_size;
  local_addr_t arcsin_coeff_local_addr = 8 * tensor_size + 3 * bank_size;

  data_type_t dtype = DT_FP32;
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_w =
      DIV_UP(MIN(length, tensor_size * npu_num / dtype_size), npu_num);

  float scalar = value.f32;
  int todo = length;
  int done = 0;
  dim4 shape = {.n = 1, .h = 1};
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  float POW_C = 2;
  scalar_t ADD_C = {.f32 = 1.f};
  scalar_t positive_zero = {.u32 = 0x00000000};
  scalar_t negtive_zero = {.u32 = 0x80000000};
  scalar_t true_val = {.u8 = 1};
  scalar_t PI_C = {.f32 = PI};
  scalar_t neg_PI_C = {.f32 = -PI};
  scalar_t half_PI_C = {.f32 = PI / 2};
  scalar_t neg_half_PI_C = {.f32 = -PI / 2};
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
  tpu_bdc_load_fp32_arcsin_coeff(arcsin_coeff_local_addr);
  while (todo != 0) {
    if (todo > NPU_NUM) {
      shape.c = NPU_NUM;
      shape.w = MIN(todo / NPU_NUM, tensor_w);
    } else {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L(input_local_addrs[index],
                     input_global_addr + done * dtype_size, &shape, NULL, NULL,
                     dtype);
    if (tpu_is_parallel_state()) {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if (l2s) {
      tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL,
                       output_dtype);
    }

    // y/x
    tpu_bdc_fp32_div_C(work_local_addrs[3], input_local_addrs[index], scalar,
                       &shape, NULL, NULL);

    // atan
    tpu_bdc_abs(work_local_addrs[2], work_local_addrs[3], &shape, NULL, NULL,
                dtype);
    tpu_bdc_fp32_pow_C(output_local_addrs[index], work_local_addrs[2],
                       work_local_addrs[0], work_local_addrs[1],
                       exp_coeff_local_addr, log_coeff_local_addr,
                       exp_table_local_addr, POW_C, &shape);
    tpu_bdc_fp_add_C(work_local_addrs[2], output_local_addrs[index], ADD_C,
                     &shape, NULL, NULL, dtype);
    tpu_bdc_fp32_rsqrt(output_local_addrs[index], work_local_addrs[2], &shape);
    tpu_bdc_fp_mul(work_local_addrs[2], output_local_addrs[index],
                   work_local_addrs[3], &shape, NULL, NULL, NULL, dtype);
    tpu_bdc_fp32_arcsin(output_local_addrs[index], work_local_addrs[2],
                        work_local_addrs[0], arcsin_coeff_local_addr, &shape);

    // y/x==-0 atan(y/x)==-0
    tpu_bdc_equal_C(work_local_addrs[0], work_local_addrs[3], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_set_C(work_local_addrs[1], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[2], negtive_zero, &shape, NULL, DT_UINT32);
    variable_t src0 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[0]}};
    variable_t src1 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[1]}};
    variable_t src2 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[2]}};
    variable_t src3 = {.type = TENSOR,
                       .context = {.addr = output_local_addrs[index]}};
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    if (scalar == 0) {
      // x==0  y > 0
      tpu_bdc_set_C(work_local_addrs[0], positive_zero, &shape, NULL, dtype);
      tpu_bdc_set_C(work_local_addrs[1], half_PI_C, &shape, NULL, dtype);
      src0.context.addr = input_local_addrs[index];
      src1.context.addr = work_local_addrs[0];
      src2.context.addr = work_local_addrs[1];
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_greater_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, dtype, dtype);
      // x==0 y < 0
      tpu_bdc_set_C(work_local_addrs[1], neg_half_PI_C, &shape, NULL, dtype);
      src3.context.addr = work_local_addrs[2];
      tpu_bdc_less_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                          &shape, dtype, dtype);

      if (value.u32 >> 31 == 0) {
        // x == +0 y==+0
        tpu_bdc_set_C(work_local_addrs[0], positive_zero, &shape, NULL,
                      DT_UINT32);
        tpu_bdc_set_C(work_local_addrs[1], positive_zero, &shape, NULL,
                      DT_UINT32);
        src3.context.addr = work_local_addrs[3];
        tpu_bdc_equal_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, DT_UINT32, dtype);

        // x == +0 y==-0
        tpu_bdc_set_C(work_local_addrs[0], negtive_zero, &shape, NULL,
                      DT_UINT32);
        tpu_bdc_set_C(work_local_addrs[1], negtive_zero, &shape, NULL,
                      DT_UINT32);
        src3.context.addr = work_local_addrs[2];
        tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                             &shape, DT_UINT32, dtype);
      } else {
        // x == -0 y==+0
        tpu_bdc_set_C(work_local_addrs[0], positive_zero, &shape, NULL,
                      DT_UINT32);
        tpu_bdc_set_C(work_local_addrs[1], PI_C, &shape, NULL, dtype);
        src3.context.addr = work_local_addrs[3];
        tpu_bdc_equal_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, DT_UINT32, dtype);

        // x == -0 y==-0
        tpu_bdc_set_C(work_local_addrs[0], negtive_zero, &shape, NULL,
                      DT_UINT32);
        tpu_bdc_set_C(work_local_addrs[1], neg_PI_C, &shape, NULL, dtype);
        src3.context.addr = work_local_addrs[2];
        tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                             &shape, DT_UINT32, dtype);
      }
    } else if (scalar < 0) {
      // x < 0   y >= 0 || y<0
      tpu_bdc_set_C(work_local_addrs[0], positive_zero, &shape, NULL, dtype);
      tpu_bdc_fp_add_C(work_local_addrs[1], work_local_addrs[3], neg_PI_C,
                       &shape, NULL, NULL, dtype);
      tpu_bdc_fp_add_C(work_local_addrs[2], work_local_addrs[3], PI_C, &shape,
                       NULL, NULL, dtype);
      src0.context.addr = input_local_addrs[index];
      src1.context.addr = work_local_addrs[0];
      src2.context.addr = work_local_addrs[1];
      src3.context.addr = work_local_addrs[2];
      tpu_bdc_less_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                          &shape, dtype, dtype);

      // x < 0 y==-0
      tpu_bdc_set_C(work_local_addrs[0], negtive_zero, &shape, NULL, DT_UINT32);
      tpu_bdc_fp_add_C(work_local_addrs[1], work_local_addrs[3], neg_PI_C,
                       &shape, NULL, NULL, dtype);
      src3.context.addr = output_local_addrs[index];
      tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                           &shape, DT_UINT32, dtype);
    }
    if (output_dtype == dtype) {
      tpu_bdc_cpy(output_local_addrs[index], work_local_addrs[3], &shape, NULL,
                  NULL, output_dtype);
    } else {
      tpu_bdc_cast(output_local_addrs[index], work_local_addrs[3], &shape, NULL,
                   NULL, output_dtype, dtype, RM_HALF_TO_EVEN);
    }

    l2s = true;
    l2s_global_addr =
        output_global_addr + done * tpu_data_type_size(output_dtype);
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }
  if (l2s) {
    tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL,
                     output_dtype);
  }
}

int tpu_kernel_api_atan2_c(const void *args) {
  sg_api_atan2_c_t *api = (sg_api_atan2_c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  scalar_t value = {.f32 = api->scalar};
  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  nodechip_atan2_c(api->input_global_addr, value, api->output_global_addr,
                   length, (data_type_t)api->dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2_c);

#ifdef BACKEND_SG2260
int tpu_kernel_api_atan2_c_multi_core(const void *args) {
  sg_api_atan2_c_t *api = (sg_api_atan2_c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  scalar_t value = {.f32 = api->scalar};
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
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }
  if (core_idx * length_slice < length) {
    nodechip_atan2_c(
        api->input_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(DT_FP32),
        value,
        api->output_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        cur_length_slice, (data_type_t)api->dtype);
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2_c_multi_core);
#endif

void nodechip_atan2(global_addr_t input_global_addr,
                    global_addr_t other_global_addr,
                    global_addr_t output_global_addr, int length,
                    data_type_t output_dtype) {
  if (length == 0) return;
  int npu_num = tpu_npu_num();
  int bank_num = tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu() / bank_num;
  int tensor_num = 2 + 2 + 2 + 4;  // 2 inputs, 2 other, 2 outputs, 4 buffer
  int coeff_bank_num = 4;          // 4 coeff
  int tensor_size = (bank_num - coeff_bank_num) / tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size > 0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t other_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t output_local_addrs[2] = {4 * tensor_size, 5 * tensor_size};
  local_addr_t work_local_addrs[4] = {6 * tensor_size, 7 * tensor_size,
                                      8 * tensor_size, 9 * tensor_size};
  local_addr_t exp_coeff_local_addr = 10 * tensor_size;
  local_addr_t log_coeff_local_addr = 10 * tensor_size + 1 * bank_size;
  local_addr_t exp_table_local_addr = 10 * tensor_size + 2 * bank_size;
  local_addr_t arcsin_coeff_local_addr = 10 * tensor_size + 3 * bank_size;

  data_type_t dtype = DT_FP32;
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_w =
      DIV_UP(MIN(length, tensor_size * npu_num / dtype_size), npu_num);

  int todo = length;
  int done = 0;
  dim4 shape = {.n = 1, .h = 1};
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  float POW_C = 2;
  scalar_t ADD_C = {.f32 = 1.f};
  scalar_t positive_zero = {.u32 = 0x00000000};
  scalar_t negtive_zero = {.u32 = 0x80000000};
  scalar_t true_val = {.u8 = 1};
  scalar_t PI_C = {.f32 = PI};
  scalar_t neg_PI_C = {.f32 = -PI};
  scalar_t half_PI_C = {.f32 = PI / 2};
  scalar_t neg_half_PI_C = {.f32 = -PI / 2};
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
  tpu_bdc_load_fp32_arcsin_coeff(arcsin_coeff_local_addr);
  while (todo != 0) {
    if (todo > NPU_NUM) {
      shape.c = NPU_NUM;
      shape.w = MIN(todo / NPU_NUM, tensor_w);
    } else {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L(input_local_addrs[index],
                     input_global_addr + done * dtype_size, &shape, NULL, NULL,
                     dtype);
    tpu_gdma_cpy_S2L(other_local_addrs[index],
                     other_global_addr + done * dtype_size, &shape, NULL, NULL,
                     dtype);
    if (tpu_is_parallel_state()) {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if (l2s) {
      tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL,
                       output_dtype);
    }

    // y/x
    tpu_bdc_fp32_div(work_local_addrs[3], input_local_addrs[index],
                     other_local_addrs[index], &shape, NULL, NULL, NULL);
    // atan
    tpu_bdc_abs(work_local_addrs[2], work_local_addrs[3], &shape, NULL, NULL,
                dtype);
    tpu_bdc_fp32_pow_C(output_local_addrs[index], work_local_addrs[2],
                       work_local_addrs[0], work_local_addrs[1],
                       exp_coeff_local_addr, log_coeff_local_addr,
                       exp_table_local_addr, POW_C, &shape);
    tpu_bdc_fp_add_C(work_local_addrs[2], output_local_addrs[index], ADD_C,
                     &shape, NULL, NULL, dtype);
    tpu_bdc_fp32_rsqrt(output_local_addrs[index], work_local_addrs[2], &shape);
    tpu_bdc_fp_mul(work_local_addrs[2], output_local_addrs[index],
                   work_local_addrs[3], &shape, NULL, NULL, NULL, dtype);
    tpu_bdc_fp32_arcsin(output_local_addrs[index], work_local_addrs[2],
                        work_local_addrs[0], arcsin_coeff_local_addr, &shape);

    // y/x==-0 atan(y/x)==-0
    tpu_bdc_equal_C(work_local_addrs[0], work_local_addrs[3], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_set_C(work_local_addrs[1], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[2], negtive_zero, &shape, NULL, DT_UINT32);
    variable_t src0 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[0]}};
    variable_t src1 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[1]}};
    variable_t src2 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[2]}};
    variable_t src3 = {.type = TENSOR,
                       .context = {.addr = output_local_addrs[index]}};
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x==+0  y==+0
    tpu_bdc_equal_C(work_local_addrs[0], other_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    DT_UINT32);
    tpu_bdc_equal_C(work_local_addrs[1], input_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    DT_UINT32);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[1], positive_zero, &shape, NULL, DT_UINT32);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x==+0 y==-0
    tpu_bdc_equal_C(work_local_addrs[0], other_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    DT_UINT32);
    tpu_bdc_equal_C(work_local_addrs[1], input_local_addrs[index], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[1], negtive_zero, &shape, NULL, DT_UINT32);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x==-0 y==+0
    tpu_bdc_equal_C(work_local_addrs[0], other_local_addrs[index], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_equal_C(work_local_addrs[1], input_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    DT_UINT32);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[1], PI_C, &shape, NULL, dtype);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x==-0 y==-0
    tpu_bdc_equal_C(work_local_addrs[0], other_local_addrs[index], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_equal_C(work_local_addrs[1], input_local_addrs[index], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[1], neg_PI_C, &shape, NULL, dtype);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x==0 y>0
    tpu_bdc_equal_C(work_local_addrs[0], other_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    dtype);
    tpu_bdc_greater_C(work_local_addrs[1], input_local_addrs[index],
                      positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                      dtype);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[1], half_PI_C, &shape, NULL, dtype);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x==0 y<0
    tpu_bdc_equal_C(work_local_addrs[0], other_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    dtype);
    tpu_bdc_less_C(work_local_addrs[1], input_local_addrs[index], positive_zero,
                   true_val, &shape, NULL, NULL, DT_UINT8, dtype);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[1], neg_half_PI_C, &shape, NULL, dtype);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x<0 y>0 || y == +0
    tpu_bdc_greater_C(work_local_addrs[1], input_local_addrs[index],
                      positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                      dtype);
    tpu_bdc_equal_C(work_local_addrs[2], input_local_addrs[index],
                    positive_zero, true_val, &shape, NULL, NULL, DT_UINT8,
                    DT_UINT32);
    tpu_bdc_or(work_local_addrs[0], work_local_addrs[1], work_local_addrs[2],
               &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_less_C(work_local_addrs[1], other_local_addrs[index], positive_zero,
                   true_val, &shape, NULL, NULL, DT_UINT8, dtype);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_fp_add_C(work_local_addrs[1], work_local_addrs[3], PI_C, &shape,
                     NULL, NULL, dtype);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    // x<0 y<0 || y==-0
    tpu_bdc_less_C(work_local_addrs[1], input_local_addrs[index], positive_zero,
                   true_val, &shape, NULL, NULL, DT_UINT8, dtype);
    tpu_bdc_equal_C(work_local_addrs[2], input_local_addrs[index], negtive_zero,
                    true_val, &shape, NULL, NULL, DT_UINT8, DT_UINT32);
    tpu_bdc_or(work_local_addrs[0], work_local_addrs[1], work_local_addrs[2],
               &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_less_C(work_local_addrs[1], other_local_addrs[index], positive_zero,
                   true_val, &shape, NULL, NULL, DT_UINT8, dtype);
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, DT_UINT8);
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL, DT_UINT8);
    tpu_bdc_fp_add_C(work_local_addrs[1], work_local_addrs[3], neg_PI_C, &shape,
                     NULL, NULL, dtype);
    src0.context.addr = work_local_addrs[2];
    src1.context.addr = work_local_addrs[0];
    src2.context.addr = work_local_addrs[1];
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, DT_UINT8, dtype);

    if (output_dtype == dtype) {
      tpu_bdc_cpy(output_local_addrs[index], work_local_addrs[3], &shape, NULL,
                  NULL, output_dtype);
    } else {
      tpu_bdc_cast(output_local_addrs[index], work_local_addrs[3], &shape, NULL,
                   NULL, output_dtype, dtype, RM_HALF_TO_EVEN);
    }

    l2s = true;
    l2s_global_addr =
        output_global_addr + done * tpu_data_type_size(output_dtype);
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }
  if (l2s) {
    tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL,
                     output_dtype);
  }
}

int tpu_kernel_api_atan2(const void *args) {
  sg_api_atan2_t *api = (sg_api_atan2_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  nodechip_atan2(api->input_global_addr, api->other_global_addr,
                 api->output_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2);

#ifdef BACKEND_SG2260
int tpu_kernel_api_atan2_multi_core(const void *args) {
  sg_api_atan2_t *api = (sg_api_atan2_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
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
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }
  if (core_idx * length_slice < length) {
    nodechip_atan2(api->input_global_addr +
                       (length_slice * core_idx) * tpu_data_type_size(DT_FP32),
                   api->other_global_addr +
                       (length_slice * core_idx) * tpu_data_type_size(DT_FP32),
                   api->output_global_addr + (length_slice * core_idx) *
                                                 tpu_data_type_size(api->dtype),
                   cur_length_slice, (data_type_t)api->dtype);
  }
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2_multi_core);
#endif

void nodechip_atan2_bcast(global_addr_t input_global_addr,
                          global_addr_t other_global_addr,
                          global_addr_t output_global_addr,
                          const dim4 *input_shape, const dim4 *other_shape,
                          const dim4 *output_shape, data_type_t output_dtype) {
  dim4 input_global_stride, other_global_stride, output_global_stride;
  tpu_continuous_stride(&input_global_stride, input_shape);
  tpu_continuous_stride(&other_global_stride, other_shape);
  tpu_continuous_stride(&output_global_stride, output_shape);

  const bool input_bcast[4] = {
      input_shape->n != output_shape->n, input_shape->c != output_shape->c,
      input_shape->h != output_shape->h, input_shape->w != output_shape->w};
  const bool other_bcast[4] = {
      other_shape->n != output_shape->n, other_shape->c != output_shape->c,
      other_shape->h != output_shape->h, other_shape->w != output_shape->w};
  bool input_bcast_all = false, other_bcast_all = false;
  for (int i = 0; i < 4; ++i) {
    input_bcast_all = input_bcast_all || input_bcast[i];
    other_bcast_all = other_bcast_all || other_bcast[i];
  }
  data_type_t dtype = DT_FP32;
  const int c_per_npu = DIV_UP(output_shape->c, NPU_NUM);
  int hmax = output_shape->h, nmax = output_shape->n,
      cmax = c_per_npu * NPU_NUM;
  local_addr_t output_addr, input_addr, other_addr;
  local_addr_t work0_addr, work1_addr, work2_addr, work3_addr, exp_coeff_addr,
      log_coeff_addr, exp_table_addr, arcsin_coeff_addr;
  while (true) {
    output_addr = 0;
    int output_size = tpu_aligned_feature_size(hmax, output_shape->w, dtype) *
                      DIV_UP(cmax, NPU_NUM) * nmax;
    input_addr = output_addr + output_size;
    int input_size = output_size;
    other_addr = input_addr + input_size;
    int other_size = output_size;
    work0_addr = other_addr + other_size;
    work1_addr = other_addr + 2 * other_size;
    work2_addr = other_addr + 3 * other_size;
    work3_addr = other_addr + 4 * other_size;
    exp_coeff_addr = other_addr + 5 * other_size;
    log_coeff_addr = other_addr + 6 * other_size;
    exp_table_addr = other_addr + 7 * other_size;
    arcsin_coeff_addr = other_addr + 8 * other_size;
    int total_size = arcsin_coeff_addr + other_size;
    if (total_size <= LOCAL_MEM_SIZE) {
      break;
    } else {
      if (cmax > NPU_NUM) {
        if (cmax % NPU_NUM == 0) {
          cmax -= NPU_NUM;
        } else {
          cmax -= (cmax % NPU_NUM);
        }
        continue;
      } else if (nmax > 1) {
        nmax /= 2;
        continue;
      } else if (hmax > 1) {
        hmax /= 2;
        continue;
      } else {
        TPUKERNEL_ASSERT(false);
      }
    }
  }
  dim4 shape = {.w = output_shape->w};
  dim4 input_local_shape, other_local_shape;
  dim4 input_local_stride, other_local_stride;
  int ctodo = output_shape->c, cdone = 0;
  float POW_C = 2;
  scalar_t ADD_C = {.f32 = 1.f};
  scalar_t positive_zero = {.u32 = 0x00000000};
  scalar_t negtive_zero = {.u32 = 0x80000000};
  scalar_t true_val = {.u8 = 1};
  scalar_t PI_C = {.f32 = PI};
  scalar_t neg_PI_C = {.f32 = -PI};
  scalar_t half_PI_C = {.f32 = PI / 2};
  scalar_t neg_half_PI_C = {.f32 = -PI / 2};
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_addr);
  tpu_bdc_load_fp32_log_coeff(log_coeff_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_addr);
  tpu_bdc_load_fp32_arcsin_coeff(arcsin_coeff_addr);

  while (ctodo > 0) {
    shape.c = MIN(ctodo, cmax);
    int ntodo = output_shape->n, ndone = 0;
    while (ntodo > 0) {
      shape.n = MIN(ntodo, nmax);
      int htodo = output_shape->h, hdone = 0;
      while (htodo > 0) {
        shape.h = MIN(htodo, hmax);
        // Move input from global memory to local memory
        tpu_aligned_stride(&input_local_stride, 0, &shape, dtype);
        input_local_shape.n = input_bcast[0] ? 1 : shape.n;
        input_local_shape.c = input_bcast[1] ? 1 : shape.c;
        input_local_shape.h = input_bcast[2] ? 1 : shape.h;
        input_local_shape.w = input_bcast[3] ? 1 : shape.w;
        global_addr_t input_global_addr_gdma =
            input_global_addr +
            ((input_bcast[0] ? 0 : ndone) * input_global_stride.n +
             (input_bcast[1] ? 0 : cdone) * input_global_stride.c +
             (input_bcast[2] ? 0 : hdone) * input_global_stride.h) *
                tpu_data_type_size(dtype);
        tpu_gdma_cpy_S2L(input_addr, input_global_addr_gdma, &input_local_shape,
                         &input_local_stride, &input_global_stride, dtype);

        // Move other from global memory to local memory
        tpu_aligned_stride(&other_local_stride, 0, &shape, dtype);
        other_local_shape.n = other_bcast[0] ? 1 : shape.n;
        other_local_shape.c = other_bcast[1] ? 1 : shape.c;
        other_local_shape.h = other_bcast[2] ? 1 : shape.h;
        other_local_shape.w = other_bcast[3] ? 1 : shape.w;
        global_addr_t other_global_addr_gdma =
            other_global_addr +
            ((other_bcast[0] ? 0 : ndone) * other_global_stride.n +
             (other_bcast[1] ? 0 : cdone) * other_global_stride.c +
             (other_bcast[2] ? 0 : hdone) * other_global_stride.h) *
                tpu_data_type_size(dtype);
        tpu_gdma_cpy_S2L(other_addr, other_global_addr_gdma, &other_local_shape,
                         &other_local_stride, &other_global_stride, dtype);

        // Broadcast input if needed
        if (input_bcast[1]) {
          input_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast(input_addr, input_addr, &input_local_shape, dtype);
        }
        if (input_bcast[0] || input_bcast[2] || input_bcast[3] ||
            (input_bcast[1] && shape.c > NPU_NUM)) {
          dim4 input_bcast_stride;
          input_bcast_stride.n = input_bcast[0] ? 0 : input_local_stride.n;
          input_bcast_stride.c = input_bcast[1] ? 0 : input_local_stride.c;
          input_bcast_stride.h = input_bcast[2] ? 0 : input_local_stride.h;
          input_bcast_stride.w = input_bcast[3] ? 0 : input_local_stride.w;
          tpu_bdc_cpy(input_addr, input_addr, &shape, NULL, &input_bcast_stride,
                      dtype);
        }

        // Broadcast other if needed
        if (other_bcast[1]) {
          other_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast(other_addr, other_addr, &other_local_shape, dtype);
        }
        if (other_bcast[0] || other_bcast[2] || other_bcast[3] ||
            (other_bcast[1] && shape.c > NPU_NUM)) {
          dim4 other_bcast_stride;
          other_bcast_stride.n = other_bcast[0] ? 0 : other_local_stride.n;
          other_bcast_stride.c = other_bcast[1] ? 0 : other_local_stride.c;
          other_bcast_stride.h = other_bcast[2] ? 0 : other_local_stride.h;
          other_bcast_stride.w = other_bcast[3] ? 0 : other_local_stride.w;
          tpu_bdc_cpy(other_addr, other_addr, &shape, NULL, &other_bcast_stride,
                      dtype);
        }

        // Select
        // y/x
        tpu_bdc_fp32_div(work3_addr, input_addr, other_addr, &shape, NULL, NULL,
                         NULL);
        // atan
        tpu_bdc_abs(work2_addr, work3_addr, &shape, NULL, NULL, dtype);
        tpu_bdc_fp32_pow_C(output_addr, work2_addr, work0_addr, work1_addr,
                           exp_coeff_addr, log_coeff_addr, exp_table_addr,
                           POW_C, &shape);
        tpu_bdc_fp_add_C(work2_addr, output_addr, ADD_C, &shape, NULL, NULL,
                         dtype);
        tpu_bdc_fp32_rsqrt(output_addr, work2_addr, &shape);
        tpu_bdc_fp_mul(work2_addr, output_addr, work3_addr, &shape, NULL, NULL,
                       NULL, dtype);
        tpu_bdc_fp32_arcsin(output_addr, work2_addr, work0_addr,
                            arcsin_coeff_addr, &shape);

        // y/x==-0 atan(y/x)==-0
        tpu_bdc_equal_C(work0_addr, work3_addr, negtive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_set_C(work1_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work2_addr, negtive_zero, &shape, NULL, DT_UINT32);
        variable_t src0 = {.type = TENSOR, .context = {.addr = work0_addr}};
        variable_t src1 = {.type = TENSOR, .context = {.addr = work1_addr}};
        variable_t src2 = {.type = TENSOR, .context = {.addr = work2_addr}};
        variable_t src3 = {.type = TENSOR, .context = {.addr = output_addr}};
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x==+0  y==+0
        tpu_bdc_equal_C(work0_addr, other_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_equal_C(work1_addr, input_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work1_addr, positive_zero, &shape, NULL, DT_UINT32);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x==+0 y==-0
        tpu_bdc_equal_C(work0_addr, other_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_equal_C(work1_addr, input_addr, negtive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work1_addr, negtive_zero, &shape, NULL, DT_UINT32);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x==-0 y==+0
        tpu_bdc_equal_C(work0_addr, other_addr, negtive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_equal_C(work1_addr, input_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work1_addr, PI_C, &shape, NULL, dtype);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x==-0 y==-0
        tpu_bdc_equal_C(work0_addr, other_addr, negtive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_equal_C(work1_addr, input_addr, negtive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work1_addr, neg_PI_C, &shape, NULL, dtype);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x==0 y>0
        tpu_bdc_equal_C(work0_addr, other_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_greater_C(work1_addr, input_addr, positive_zero, true_val,
                          &shape, NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work1_addr, half_PI_C, &shape, NULL, dtype);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x==0 y<0
        tpu_bdc_equal_C(work0_addr, other_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_less_C(work1_addr, input_addr, positive_zero, true_val, &shape,
                       NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_set_C(work1_addr, neg_half_PI_C, &shape, NULL, dtype);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x<0 y>0 || y == +0
        tpu_bdc_greater_C(work1_addr, input_addr, positive_zero, true_val,
                          &shape, NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_equal_C(work2_addr, input_addr, positive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_or(work0_addr, work1_addr, work2_addr, &shape, NULL, NULL, NULL,
                   DT_UINT8);
        tpu_bdc_less_C(work1_addr, other_addr, positive_zero, true_val, &shape,
                       NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_fp_add_C(work1_addr, work3_addr, PI_C, &shape, NULL, NULL,
                         dtype);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        // x<0 y<0 || y==-0
        tpu_bdc_less_C(work1_addr, input_addr, positive_zero, true_val, &shape,
                       NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_equal_C(work2_addr, input_addr, negtive_zero, true_val, &shape,
                        NULL, NULL, DT_UINT8, DT_UINT32);
        tpu_bdc_or(work0_addr, work1_addr, work2_addr, &shape, NULL, NULL, NULL,
                   DT_UINT8);
        tpu_bdc_less_C(work1_addr, other_addr, positive_zero, true_val, &shape,
                       NULL, NULL, DT_UINT8, dtype);
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, DT_UINT8);
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL, DT_UINT8);
        tpu_bdc_fp_add_C(work1_addr, work3_addr, neg_PI_C, &shape, NULL, NULL,
                         dtype);
        src0.context.addr = work2_addr;
        src1.context.addr = work0_addr;
        src2.context.addr = work1_addr;
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             DT_UINT8, dtype);

        if (output_dtype == dtype) {
          tpu_bdc_cpy(output_addr, work3_addr, &shape, NULL, NULL,
                      output_dtype);
        } else {
          tpu_bdc_cast(output_addr, work3_addr, &shape, NULL, NULL,
                       output_dtype, dtype, RM_HALF_TO_EVEN);
        }

        // Move out from local memory to global memory
        global_addr_t output_global_addr_gdma =
            output_global_addr +
            (ndone * output_global_stride.n + cdone * output_global_stride.c +
             hdone * output_global_stride.h) *
                tpu_data_type_size(output_dtype);
        tpu_gdma_cpy_L2S(output_global_addr_gdma, output_addr, &shape,
                         &output_global_stride, NULL, output_dtype);
        htodo -= shape.h;
        hdone += shape.h;
      }
      ntodo -= shape.n;
      ndone += shape.n;
    }
    ctodo -= shape.c;
    cdone += shape.c;
  }
}

int tpu_kernel_api_atan2_bcast(const void *args) {
  sg_api_atan2_bcast_t *api = (sg_api_atan2_bcast_t *)args;
  TPUKERNEL_ASSERT(api->output_dim > 0 && api->output_dim <= 4);

  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  dim4 input_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
  dim4 other_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
  dim4 output_shape = {.n = 1, .c = 1, .h = 1, .w = 1};

  if (api->output_dim >= 1) {
    if (api->input_dim >= 1)
      input_shape.w = api->input_shape[api->input_dim - 1];
    if (api->other_dim >= 1)
      other_shape.w = api->other_shape[api->other_dim - 1];
    output_shape.w =
        input_shape.w > other_shape.w ? input_shape.w : other_shape.w;
  }
  if (api->output_dim >= 2) {
    if (api->input_dim >= 2)
      input_shape.h = api->input_shape[api->input_dim - 2];
    if (api->other_dim >= 2)
      other_shape.h = api->other_shape[api->other_dim - 2];
    output_shape.h =
        input_shape.h > other_shape.h ? input_shape.h : other_shape.h;
  }
  if (api->output_dim >= 3) {
    if (api->input_dim >= 3)
      input_shape.c = api->input_shape[api->input_dim - 3];
    if (api->other_dim >= 3)
      other_shape.c = api->other_shape[api->other_dim - 3];
    output_shape.c =
        input_shape.c > other_shape.c ? input_shape.c : other_shape.c;
  }
  if (api->output_dim >= 4) {
    if (api->input_dim >= 4)
      input_shape.n = api->input_shape[api->input_dim - 4];
    if (api->other_dim >= 4)
      other_shape.n = api->other_shape[api->other_dim - 4];
    output_shape.n =
        input_shape.n > other_shape.n ? input_shape.n : other_shape.n;
  }

  tpu_initialize();
  nodechip_atan2_bcast(api->input_global_addr, api->other_global_addr,
                       api->output_global_addr, &input_shape, &other_shape,
                       &output_shape, (data_type_t)api->dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2_bcast);

#ifdef BACKEND_SG2260
int tpu_kernel_api_atan2_bcast_multi_core(const void *args) {
  sg_api_atan2_bcast_t *api = (sg_api_atan2_bcast_t *)args;
  TPUKERNEL_ASSERT(api->output_dim > 0 && api->output_dim <= 4);

  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);

  dim4 input_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
  dim4 other_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
  dim4 output_shape = {.n = 1, .c = 1, .h = 1, .w = 1};

  if (api->output_dim >= 1) {
    if (api->input_dim >= 1)
      input_shape.w = api->input_shape[api->input_dim - 1];
    if (api->other_dim >= 1)
      other_shape.w = api->other_shape[api->other_dim - 1];
    output_shape.w =
        input_shape.w > other_shape.w ? input_shape.w : other_shape.w;
  }
  if (api->output_dim >= 2) {
    if (api->input_dim >= 2)
      input_shape.h = api->input_shape[api->input_dim - 2];
    if (api->other_dim >= 2)
      other_shape.h = api->other_shape[api->other_dim - 2];
    output_shape.h =
        input_shape.h > other_shape.h ? input_shape.h : other_shape.h;
  }
  if (api->output_dim >= 3) {
    if (api->input_dim >= 3)
      input_shape.c = api->input_shape[api->input_dim - 3];
    if (api->other_dim >= 3)
      other_shape.c = api->other_shape[api->other_dim - 3];
    output_shape.c =
        input_shape.c > other_shape.c ? input_shape.c : other_shape.c;
  }
  if (api->output_dim >= 4) {
    if (api->input_dim >= 4)
      input_shape.n = api->input_shape[api->input_dim - 4];
    if (api->other_dim >= 4)
      other_shape.n = api->other_shape[api->other_dim - 4];
    output_shape.n =
        input_shape.n > other_shape.n ? input_shape.n : other_shape.n;
  }

  int length = MAX(input_shape.n, other_shape.n);
  tpu_initialize();
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();

  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }

  int input_offset = (input_shape.n != 1 ? length_slice : 0) * core_idx *
                     input_shape.c * input_shape.h * input_shape.w *
                     tpu_data_type_size(DT_FP32);
  int other_offset = (other_shape.n != 1 ? length_slice : 0) * core_idx *
                     other_shape.c * other_shape.h * other_shape.w *
                     tpu_data_type_size(DT_FP32);
  int output_offset = (output_shape.n != 1 ? length_slice : 0) * core_idx *
                      output_shape.c * output_shape.h * output_shape.w *
                      tpu_data_type_size(api->dtype);

  input_shape.n = input_shape.n != 1 ? cur_length_slice : 1;
  other_shape.n = other_shape.n != 1 ? cur_length_slice : 1;
  output_shape.n = MAX(input_shape.n, other_shape.n);

  if (core_idx * length_slice < length) {
    nodechip_atan2_bcast(api->input_global_addr + input_offset,
                         api->other_global_addr + other_offset,
                         api->output_global_addr + output_offset, &input_shape,
                         &other_shape, &output_shape, (data_type_t)api->dtype);
  }

  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_atan2_bcast_multi_core);
#endif