#include "kernel_utils_func.h"
#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

void nodechip_nextafterc(float scalar, global_addr_t other_global_addr,
                         global_addr_t output_global_addr, int length,
                         data_type_t dtype) {
  if (length == 0) return;
  int npu_num = tpu_npu_num();
  int bank_num = tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu() / bank_num;
  int tensor_num = 2 + 2 + 4;  // 2 inputs, 2 outputs, 4 buffer
  int coeff_bank_num = 0;      // 0 coeff
  int tensor_size = (bank_num - coeff_bank_num) / tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size > 0);

  local_addr_t other_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t work_local_addrs[4] = {4 * tensor_size, 5 * tensor_size,
                                      6 * tensor_size, 7 * tensor_size};

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

  scalar_t positive_C;
  scalar_t negtive_C;
  scalar_t positive_one_C;

  if (dtype == DT_FP32) {
    positive_C.u32 = 0x00000001;
    negtive_C.u32 = 0x80000001;
    positive_one_C.s32 = 1;
  } else {
    positive_C.u16 = 0x0001;
    negtive_C.u16 = 0x8001;
    positive_one_C.s16 = 1;
  }

  scalar_t value = {.f32 = scalar};
  scalar_t scalar_C = tpu_cast(value, dtype, DT_FP32, RM_HALF_TO_EVEN);
  data_type_t dtype_int = dtype == DT_FP32 ? DT_INT32 : DT_INT16;
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
                       dtype);
    }
    // input == nan -> output = nan
    if ((dtype == DT_FP32 && ((scalar_C.u32 & 0x7f800000) == 0x7f800000) &&
         ((scalar_C.u32 & 0x7fffffff) != 0x7f800000)) ||
        (dtype == DT_BFP16 && ((scalar_C.u16 & 0x7f80) == 0x7f80) &&
         ((scalar_C.u16 & 0x7fff) != 0x7f80))) {
      tpu_bdc_set_C(output_local_addrs[index], scalar_C, &shape, NULL, dtype);
      tpu_print_local_mem_data(output_local_addrs[index], 0, &shape, NULL,
                               dtype);
    } else if (scalar > 0) {
      tpu_bdc_set_C(work_local_addrs[0], scalar_C, &shape, NULL,
                    dtype);  // work0 input
      tpu_bdc_int_add_C(work_local_addrs[1], work_local_addrs[0],
                        positive_one_C, &shape, NULL, NULL, dtype_int,
                        dtype_int, dtype_int, 0, 0, true);  // work1 input+1
      tpu_bdc_set_C(work_local_addrs[2], scalar_C, &shape, NULL,
                    dtype);  // work2 input
      // work3 = work0 < other ? work1 : work2
      variable_t src0 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[0]}};
      variable_t src1 = {.type = TENSOR,
                         .context = {.addr = other_local_addrs[index]}};
      variable_t src2 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[1]}};
      variable_t src3 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[2]}};
      tpu_bdc_less_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                          &shape, dtype, dtype_int);

      tpu_bdc_int_sub_C(work_local_addrs[1], work_local_addrs[0],
                        positive_one_C, &shape, NULL, NULL, dtype_int,
                        dtype_int, dtype_int, 0, 0,
                        true);  // work1 input-1
      // work2 = work0 > other ? work1 : work3
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_greater_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, dtype, dtype_int);

      // other == Nan -> output = nan
      isNan(work_local_addrs[0], other_local_addrs[index], work_local_addrs[1],
            work_local_addrs[3], &shape, NULL, dtype);
      replaceWithNan(output_local_addrs[index], work_local_addrs[2],
                     work_local_addrs[0], work_local_addrs[1],
                     work_local_addrs[3], &shape, NULL, dtype);

    } else if (scalar < 0) {
      tpu_bdc_set_C(work_local_addrs[0], scalar_C, &shape, NULL,
                    dtype);  // work0 input
      tpu_bdc_int_sub_C(work_local_addrs[1], work_local_addrs[0],
                        positive_one_C, &shape, NULL, NULL, dtype_int,
                        dtype_int, dtype_int, 0, 0,
                        true);  // work1 input-1
      tpu_bdc_set_C(work_local_addrs[2], scalar_C, &shape, NULL,
                    dtype);  // work2 input
      // work3 = work0 < other ? work1 : work2
      variable_t src0 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[0]}};
      variable_t src1 = {.type = TENSOR,
                         .context = {.addr = other_local_addrs[index]}};
      variable_t src2 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[1]}};
      variable_t src3 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[2]}};
      tpu_bdc_less_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                          &shape, dtype, dtype_int);
      tpu_bdc_int_add_C(work_local_addrs[1], work_local_addrs[0],
                        positive_one_C, &shape, NULL, NULL, dtype_int,
                        dtype_int, dtype_int, 0, 0, true);  // work1 input+1
      // work2 = work0 > other ? work1 : work3
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_greater_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, dtype, dtype_int);

      // other == Nan -> output = nan
      isNan(work_local_addrs[0], other_local_addrs[index], work_local_addrs[1],
            work_local_addrs[3], &shape, NULL, dtype);
      replaceWithNan(output_local_addrs[index], work_local_addrs[2],
                     work_local_addrs[0], work_local_addrs[1],
                     work_local_addrs[3], &shape, NULL, dtype);

    } else if (scalar == 0) {
      tpu_bdc_set_C(work_local_addrs[0], scalar_C, &shape, NULL,
                    dtype);  // work0 input
      tpu_bdc_set_C(work_local_addrs[1], positive_C, &shape, NULL,
                    dtype_int);  // work1 (0x00000001 or 0x0001)
      // tpu_bdc_set_C(work_local_addrs[2], scalar_C, &shape, NULL,
      //               dtype);  // work2 input
      // work3 = work0 < other ? work1 : other
      variable_t src0 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[0]}};
      variable_t src1 = {.type = TENSOR,
                         .context = {.addr = other_local_addrs[index]}};
      variable_t src2 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[1]}};
      variable_t src3 = {.type = TENSOR,
                         .context = {.addr = other_local_addrs[index]}};
      tpu_bdc_less_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                          &shape, dtype, dtype_int);

      tpu_bdc_set_C(work_local_addrs[1], negtive_C, &shape, NULL,
                    dtype_int);  // work1 (0x80000001 or 0x8001)
      // work2 = work0 > other ? work1 : other
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_greater_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, dtype, dtype_int);

      // other == Nan -> output = nan
      isNan(work_local_addrs[0], other_local_addrs[index], work_local_addrs[1],
            work_local_addrs[3], &shape, NULL, dtype);
      replaceWithNan(output_local_addrs[index], work_local_addrs[2],
                     work_local_addrs[0], work_local_addrs[1],
                     work_local_addrs[3], &shape, NULL, dtype);

      tpu_print_local_mem_data(output_local_addrs[index], 0, &shape, NULL,
                               dtype_int);
    }
    l2s = true;
    l2s_global_addr = output_global_addr + done * dtype_size;
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
                     dtype);
  }
}

void tpu_kernel_api_nextafterc(const void *args) {
  sg_api_nextafterc_t *api = (sg_api_nextafterc_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  nodechip_nextafterc(api->scalar, api->other_global_addr,
                      api->output_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafterc);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_nextafterc_multi_core(const void *args) {
  sg_api_nextafterc_t *api = (sg_api_nextafterc_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

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
    nodechip_nextafterc(
        api->scalar,
        api->other_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        api->output_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        cur_length_slice, (data_type_t)api->dtype);
  }

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafterc_multi_core);
#endif

void nodechip_nextafter_c(global_addr_t input_global_addr, float scalar,
                          global_addr_t output_global_addr, int length,
                          data_type_t dtype) {
  if (length == 0) return;
  int npu_num = tpu_npu_num();
  int bank_num = tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu() / bank_num;
  int tensor_num = 2 + 2 + 3;  // 2 inputs, 2 outputs, 4 buffer
  int coeff_bank_num = 0;      // 0 coeff
  int tensor_size = (bank_num - coeff_bank_num) / tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size > 0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t work_local_addrs[4] = {4 * tensor_size, 5 * tensor_size,
                                      6 * tensor_size, 7 * tensor_size};

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

  scalar_t value = {.f32 = scalar};
  scalar_t scalar_C = tpu_cast(value, dtype, DT_FP32, RM_HALF_TO_EVEN);
  scalar_t true_val;
  data_type_t dtype_int = dtype == DT_FP32 ? DT_INT32 : DT_INT16;
  data_type_t dtype_uint = dtype == DT_FP32 ? DT_UINT32 : DT_UINT16;
  scalar_t positive_zero_C;
  // scalar_t negtive_zero_C;
  scalar_t positive_one_C;
  scalar_t positive_C;
  scalar_t negtive_C;

  if (dtype == DT_FP32) {
    positive_zero_C.u32 = 0x00000000;
    // negtive_zero_C.u32 = 0x80000000;
    positive_C.u32 = 0x00000001;
    negtive_C.u32 = 0x80000001;
    positive_one_C.s32 = 1;
    true_val.u32 = 1;
  } else {
    positive_zero_C.u16 = 0x0000;
    // negtive_zero_C.u16 = 0x8000;
    positive_C.u16 = 0x0001;
    negtive_C.u16 = 0x8001;
    positive_one_C.s16 = 1;
    true_val.u16 = 1;
  }

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
                       dtype);
    }

    // scalar == nan -> output = nan
    if ((dtype == DT_FP32 && ((scalar_C.u32 & 0x7f800000) == 0x7f800000) &&
         ((scalar_C.u32 & 0x7fffffff) != 0x7f800000)) ||
        (dtype == DT_BFP16 && ((scalar_C.u16 & 0x7f80) == 0x7f80) &&
         ((scalar_C.u16 & 0x7fff) != 0x7f80))) {
      tpu_bdc_set_C(output_local_addrs[index], scalar_C, &shape, NULL, dtype);
    } else {
      // >0 >s
      tpu_bdc_greater_C(work_local_addrs[0], input_local_addrs[index],
                        positive_zero_C, true_val, &shape, NULL, NULL,
                        dtype_uint,
                        dtype);  // work0 >0
      tpu_bdc_greater_C(work_local_addrs[1], input_local_addrs[index], scalar_C,
                        true_val, &shape, NULL, NULL, dtype_uint,
                        dtype);  // work1 >s
      tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                  &shape, NULL, NULL, NULL, dtype_uint);  // work2 >0 && >s
      tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                    dtype_uint);  // work0 true_val
      tpu_bdc_int_sub_C(work_local_addrs[1], input_local_addrs[index],
                        positive_one_C, &shape, NULL, NULL, dtype_int,
                        dtype_int, dtype_int, 0, 0, true);  // work1 input-1
      // work3 =  work2 == work0 ? work1 : input
      variable_t src0 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[2]}};
      variable_t src1 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[0]}};
      variable_t src2 = {.type = TENSOR,
                         .context = {.addr = work_local_addrs[1]}};
      variable_t src3 = {.type = TENSOR,
                         .context = {.addr = input_local_addrs[index]}};
      tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                           &shape, dtype_uint, dtype);

      // <0 <s
      tpu_bdc_less_C(work_local_addrs[0], input_local_addrs[index],
                     positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                     dtype);  // work0 <0
      tpu_bdc_less_C(output_local_addrs[index], input_local_addrs[index],
                     scalar_C, true_val, &shape, NULL, NULL, dtype_uint,
                     dtype);  // output <s
      tpu_bdc_and(work_local_addrs[2], work_local_addrs[0],
                  output_local_addrs[index], &shape, NULL, NULL, NULL,
                  dtype_uint);  // work2 <0 && <s
      tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                    dtype_uint);  // work0 true_val
      // output = work2 == work0 ? work1 : work3
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2,
                           &src3, &shape, dtype_uint, dtype);

      // >0 <s
      tpu_bdc_greater_C(work_local_addrs[0], input_local_addrs[index],
                        positive_zero_C, true_val, &shape, NULL, NULL,
                        dtype_uint,
                        dtype);  // work0 >0
      tpu_bdc_less_C(work_local_addrs[1], input_local_addrs[index], scalar_C,
                     true_val, &shape, NULL, NULL, dtype_uint,
                     dtype);  // work1 <s
      tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                  &shape, NULL, NULL, NULL, dtype_uint);  // work2 >0 && <s
      tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                    dtype_uint);  // work0 true_val
      tpu_bdc_int_add_C(work_local_addrs[1], input_local_addrs[index],
                        positive_one_C, &shape, NULL, NULL, dtype_int,
                        dtype_int, dtype_int, 0, 0, true);  // work1 input+1
      // work3 = work2 == work0 ? work1 : output
      src3.context.addr = output_local_addrs[index];
      tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                           &shape, dtype_uint, dtype);

      // <0 >s
      tpu_bdc_less_C(work_local_addrs[0], input_local_addrs[index],
                     positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                     dtype);  // work0 <0
      tpu_bdc_greater_C(output_local_addrs[index], input_local_addrs[index],
                        scalar_C, true_val, &shape, NULL, NULL, dtype_uint,
                        dtype);  // work1 >s
      tpu_bdc_and(work_local_addrs[2], work_local_addrs[0],
                  output_local_addrs[index], &shape, NULL, NULL, NULL,
                  dtype_uint);  // work2 <0 && >s
      tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                    dtype_uint);  // work0 true_val
      // work3 = work2 == work0 ? work1 : work3
      src3.context.addr = work_local_addrs[3];
      tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2,
                           &src3, &shape, dtype_uint, dtype);

      // input ==nan -> output = nan
      isNan(work_local_addrs[0], input_local_addrs[index], work_local_addrs[1],
            work_local_addrs[2], &shape, NULL, dtype);
      replaceWithNan(work_local_addrs[3], output_local_addrs[index],
                     work_local_addrs[0], work_local_addrs[1],
                     work_local_addrs[2], &shape, NULL, dtype);
      tpu_bdc_cpy(output_local_addrs[index], work_local_addrs[3], &shape, NULL,
                  NULL, dtype);

      // input==0 && s<0  -> output = 0x80000001 or 0x8001
      // input==0 && s>0  -> output = 0x00000001 or 0x0001
      if (scalar < 0) {
        tpu_bdc_set_C(work_local_addrs[0], positive_zero_C, &shape, NULL,
                      dtype);  // work0 0
        tpu_bdc_set_C(work_local_addrs[1], negtive_C, &shape, NULL,
                      dtype_int);  // work1 0x80000001 or 0x8001
        src0.context.addr = input_local_addrs[index];
        src1.context.addr = work_local_addrs[0];
        src2.context.addr = work_local_addrs[1];
        src3.context.addr = output_local_addrs[index];
        tpu_bdc_equal_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, dtype, dtype_uint);
        tpu_bdc_cpy(output_local_addrs[index], work_local_addrs[2], &shape,
                    NULL, NULL, dtype);
      } else if (scalar > 0) {
        tpu_bdc_set_C(work_local_addrs[0], positive_zero_C, &shape, NULL,
                      dtype);  // work0 0
        tpu_bdc_set_C(work_local_addrs[1], positive_C, &shape, NULL,
                      dtype_int);  // work1 0x00000001 or 0x0001
        src0.context.addr = input_local_addrs[index];
        src1.context.addr = work_local_addrs[0];
        src2.context.addr = work_local_addrs[1];
        src3.context.addr = output_local_addrs[index];
        tpu_bdc_equal_select(work_local_addrs[2], &src0, &src1, &src2, &src3,
                             &shape, dtype, dtype_uint);
        tpu_bdc_cpy(output_local_addrs[index], work_local_addrs[2], &shape,
                    NULL, NULL, dtype);
      }
    }
    l2s = true;
    l2s_global_addr = output_global_addr + done * dtype_size;
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
                     dtype);
  }
}

void tpu_kernel_api_nextafter_c(const void *args) {
  sg_api_nextafter_c_t *api = (sg_api_nextafter_c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  nodechip_nextafter_c(api->input_global_addr, api->scalar,
                       api->output_global_addr, length,
                       (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafter_c);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_nextafter_c_multi_core(const void *args) {
  sg_api_nextafter_c_t *api = (sg_api_nextafter_c_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

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
    nodechip_nextafter_c(
        api->input_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        api->scalar,
        api->output_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        cur_length_slice, (data_type_t)api->dtype);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafter_c_multi_core);
#endif

void nodechip_nextafter(global_addr_t input_global_addr,
                        global_addr_t other_global_addr,
                        global_addr_t output_global_addr, int length,
                        data_type_t dtype) {
  if (length == 0) return;
  int npu_num = tpu_npu_num();
  int bank_num = tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu() / bank_num;
  int tensor_num = 2 + 2 + 2 + 4;  // 2 inputs, 2 other, 2 outputs, 4 buffer
  int coeff_bank_num = 0;          // 0 coeff
  int tensor_size = (bank_num - coeff_bank_num) / tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size > 0);

  local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
  local_addr_t other_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
  local_addr_t output_local_addrs[2] = {4 * tensor_size, 5 * tensor_size};
  local_addr_t work_local_addrs[4] = {6 * tensor_size, 7 * tensor_size,
                                      8 * tensor_size, 9 * tensor_size};

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

  data_type_t dtype_int = dtype == DT_FP32 ? DT_INT32 : DT_INT16;
  data_type_t dtype_uint = dtype == DT_FP32 ? DT_UINT32 : DT_UINT16;
  scalar_t positive_zero_C;
  scalar_t positive_one_C;
  scalar_t true_val;
  scalar_t positive_C;
  scalar_t negtive_C;

  if (dtype == DT_FP32) {
    positive_zero_C.u32 = 0x00000000;
    positive_C.u32 = 0x00000001;
    negtive_C.u32 = 0x80000001;
    positive_one_C.s32 = 1;
    true_val.u32 = 1;
  } else {
    positive_zero_C.u16 = 0x0000;
    positive_C.u16 = 0x0001;
    negtive_C.u16 = 0x8001;
    positive_one_C.s16 = 1;
    true_val.u16 = 1;
  }

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
                       dtype);
    }

    // >0 >other
    tpu_bdc_greater_C(work_local_addrs[0], input_local_addrs[index],
                      positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                      dtype);  // work0 >0
    tpu_bdc_greater(work_local_addrs[1], input_local_addrs[index],
                    other_local_addrs[index], true_val, &shape, NULL, NULL,
                    NULL, dtype_uint, dtype);  // work1 >other
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, dtype_uint);  // work2 >0 && >other
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    tpu_bdc_int_sub_C(work_local_addrs[1], input_local_addrs[index],
                      positive_one_C, &shape, NULL, NULL, dtype_int, dtype_int,
                      dtype_int, 0, 0, true);  // work1 input-1
    // work3 =  work2 == work0 ? work1 : input
    variable_t src0 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[2]}};
    variable_t src1 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[0]}};
    variable_t src2 = {.type = TENSOR,
                       .context = {.addr = work_local_addrs[1]}};
    variable_t src3 = {.type = TENSOR,
                       .context = {.addr = input_local_addrs[index]}};
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    // <0 <other
    tpu_bdc_less_C(work_local_addrs[0], input_local_addrs[index],
                   positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                   dtype);  // work0 <0
    tpu_bdc_less(output_local_addrs[index], input_local_addrs[index],
                 other_local_addrs[index], true_val, &shape, NULL, NULL, NULL,
                 dtype_uint, dtype);  // output <other
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0],
                output_local_addrs[index], &shape, NULL, NULL, NULL,
                dtype_uint);  // work2 <0 && <other
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    // output = work2 == work0 ? work1 : work3
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    // >0 <other
    tpu_bdc_greater_C(work_local_addrs[0], input_local_addrs[index],
                      positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                      dtype);  // work0 >0
    tpu_bdc_less(work_local_addrs[1], input_local_addrs[index],
                 other_local_addrs[index], true_val, &shape, NULL, NULL, NULL,
                 dtype_uint, dtype);  // work1 <other
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL, dtype_uint);  // work2 >0 && <other
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    tpu_bdc_int_add_C(work_local_addrs[1], input_local_addrs[index],
                      positive_one_C, &shape, NULL, NULL, dtype_int, dtype_int,
                      dtype_int, 0, 0, true);  // work1 input+1
    // work3 = work2 == work0 ? work1 : output
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    // <0 >other
    tpu_bdc_less_C(work_local_addrs[0], input_local_addrs[index],
                   positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                   dtype);  // work0 <0
    tpu_bdc_greater(output_local_addrs[index], input_local_addrs[index],
                    other_local_addrs[index], true_val, &shape, NULL, NULL,
                    NULL, dtype_uint, dtype);  // work1 >other
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0],
                output_local_addrs[index], &shape, NULL, NULL, NULL,
                dtype_uint);  // work2 <0 && >other
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    // output = work2 == work0 ? work1 : work3
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    // input==0 && other==0 -> output = 0
    tpu_bdc_equal_C(work_local_addrs[0], input_local_addrs[index],
                    positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                    dtype);  // work0 input==0
    tpu_bdc_equal_C(work_local_addrs[1], other_local_addrs[index],
                    positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                    dtype);  // work1 other==0
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL,
                dtype_uint);  // work2 input==0 && other==0
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    tpu_bdc_set_C(work_local_addrs[1], positive_zero_C, &shape, NULL,
                  dtype);  // work1 0
    // work3 = work2 == work0 ? work1 : output
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    // input==0 && other > 0 -> output = 0x00000001 or 0x0001
    tpu_bdc_equal_C(work_local_addrs[0], input_local_addrs[index],
                    positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                    dtype);  // work0 input==0
    tpu_bdc_greater_C(work_local_addrs[1], other_local_addrs[index],
                      positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                      dtype);  // work1 other>0
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL,
                dtype_uint);  // work2 input==0 && other>0
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    tpu_bdc_set_C(work_local_addrs[1], positive_C, &shape, NULL,
                  dtype_uint);  // work1 0x00000001 or 0x0001
    // output = work2 == work0 ? work1 : work3
    src3.context.addr = work_local_addrs[3];
    tpu_bdc_equal_select(output_local_addrs[index], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    // input==0 && other < 0 -> output = 0x80000001 or 0x8001
    tpu_bdc_equal_C(work_local_addrs[0], input_local_addrs[index],
                    positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                    dtype);  // work0 input==0
    tpu_bdc_less_C(work_local_addrs[1], other_local_addrs[index],
                   positive_zero_C, true_val, &shape, NULL, NULL, dtype_uint,
                   dtype);  // work1 other<0
    tpu_bdc_and(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
                &shape, NULL, NULL, NULL,
                dtype_uint);  // work2 input==0 && other<0
    tpu_bdc_set_C(work_local_addrs[0], true_val, &shape, NULL,
                  dtype_uint);  // work0 true_val
    tpu_bdc_set_C(work_local_addrs[1], negtive_C, &shape, NULL,
                  dtype_uint);  // work1 0x80000001 or 0x8001
    // work3 = work2 == work0 ? work1 : output
    src3.context.addr = output_local_addrs[index];
    tpu_bdc_equal_select(work_local_addrs[3], &src0, &src1, &src2, &src3,
                         &shape, dtype_uint, dtype);

    isNan(work_local_addrs[0], input_local_addrs[index], work_local_addrs[1],
          work_local_addrs[2], &shape, NULL, dtype);
    isNan(work_local_addrs[1], other_local_addrs[index], work_local_addrs[2],
          output_local_addrs[index], &shape, NULL, dtype);
    tpu_bdc_or(work_local_addrs[2], work_local_addrs[0], work_local_addrs[1],
               &shape, NULL, NULL, NULL, dtype_uint);
    replaceWithNan(output_local_addrs[index], work_local_addrs[3],
                   work_local_addrs[2], work_local_addrs[0],
                   work_local_addrs[1], &shape, NULL, dtype);

    l2s = true;
    l2s_global_addr = output_global_addr + done * dtype_size;
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
                     dtype);
  }
}

void tpu_kernel_api_nextafter(const void *args) {
  sg_api_nextafter_t *api = (sg_api_nextafter_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

  int length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  tpu_initialize();
  nodechip_nextafter(api->input_global_addr, api->other_global_addr,
                     api->output_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafter);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_nextafter_multi_core(const void *args) {
  sg_api_nextafter_t *api = (sg_api_nextafter_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

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
    nodechip_nextafter(
        api->input_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        api->other_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        api->output_global_addr +
            (length_slice * core_idx) * tpu_data_type_size(api->dtype),
        cur_length_slice, (data_type_t)api->dtype);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafter_multi_core);
#endif

void nodechip_nextafter_bcast(global_addr_t input_global_addr,
                              global_addr_t other_global_addr,
                              global_addr_t output_global_addr,
                              const dim4 *input_shape, const dim4 *other_shape,
                              const dim4 *output_shape, data_type_t dtype) {
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

  const int c_per_npu = DIV_UP(output_shape->c, NPU_NUM);
  int hmax = output_shape->h, nmax = output_shape->n,
      cmax = c_per_npu * NPU_NUM;
  local_addr_t output_addr, input_addr, other_addr;
  local_addr_t work0_addr, work1_addr, work2_addr, work3_addr;
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
    int total_size = work3_addr + other_size;
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

  data_type_t dtype_int = dtype == DT_FP32 ? DT_INT32 : DT_INT16;
  data_type_t dtype_uint = dtype == DT_FP32 ? DT_UINT32 : DT_UINT16;
  scalar_t positive_zero_C;
  scalar_t positive_one_C;
  scalar_t positive_C;
  scalar_t negtive_C;
  scalar_t true_val;

  if (dtype == DT_FP32) {
    positive_zero_C.u32 = 0x00000000;
    positive_C.u32 = 0x00000001;
    negtive_C.u32 = 0x80000001;
    positive_one_C.s32 = 1;
    true_val.u32 = 1;
  } else {
    positive_zero_C.u16 = 0x0000;
    positive_C.u16 = 0x0001;
    negtive_C.u16 = 0x8001;
    positive_one_C.s16 = 1;
    true_val.u16 = 1;
  }

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

        // >0 >other
        tpu_bdc_greater_C(work0_addr, input_addr, positive_zero_C, true_val,
                          &shape, NULL, NULL, dtype_uint,
                          dtype);  // work0 >0
        tpu_bdc_greater(work1_addr, input_addr, other_addr, true_val, &shape,
                        NULL, NULL, NULL, dtype_uint, dtype);  // work1 >other
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, dtype_uint);  // work2 >0 && >other
        tpu_bdc_set_C(work0_addr, positive_one_C, &shape, NULL,
                      dtype_uint);  // work0 true_val
        tpu_bdc_int_sub_C(work1_addr, input_addr, positive_one_C, &shape, NULL,
                          NULL, dtype_int, dtype_int, dtype_int, 0, 0,
                          true);  // work1 input-1
        // work3 =  work2 == work0 ? work1 : input
        variable_t src0 = {.type = TENSOR, .context = {.addr = work2_addr}};
        variable_t src1 = {.type = TENSOR, .context = {.addr = work0_addr}};
        variable_t src2 = {.type = TENSOR, .context = {.addr = work1_addr}};
        variable_t src3 = {.type = TENSOR, .context = {.addr = input_addr}};
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        // <0 <other
        tpu_bdc_less_C(work0_addr, input_addr, positive_zero_C, true_val,
                       &shape, NULL, NULL, dtype_uint,
                       dtype);  // work0 <0
        tpu_bdc_less(output_addr, input_addr, other_addr, positive_one_C,
                     &shape, NULL, NULL, NULL, dtype_uint,
                     dtype);  // output <other
        tpu_bdc_and(work2_addr, work0_addr, output_addr, &shape, NULL, NULL,
                    NULL, dtype_uint);  // work2 <0 && <other
        tpu_bdc_set_C(work0_addr, positive_one_C, &shape, NULL,
                      dtype_uint);  // work0 true_val
        // output = work2 == work0 ? work1 : work3
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        // >0 <other
        tpu_bdc_greater_C(work0_addr, input_addr, positive_zero_C, true_val,
                          &shape, NULL, NULL, dtype_uint,
                          dtype);  // work0 >0
        tpu_bdc_less(work1_addr, input_addr, other_addr, true_val, &shape, NULL,
                     NULL, NULL, dtype_uint, dtype);  // work1 <other
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL, dtype_uint);  // work2 >0 && <other
        tpu_bdc_set_C(work0_addr, positive_one_C, &shape, NULL,
                      dtype_uint);  // work0 true_val
        tpu_bdc_int_add_C(work1_addr, input_addr, positive_one_C, &shape, NULL,
                          NULL, dtype_int, dtype_int, dtype_int, 0, 0,
                          true);  // work1 input+1
        // work3 = work2 == work0 ? work1 : output
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        // <0 >other
        tpu_bdc_less_C(work0_addr, input_addr, positive_zero_C, true_val,
                       &shape, NULL, NULL, dtype_uint,
                       dtype);  // work0 <0
        tpu_bdc_greater(output_addr, input_addr, other_addr, true_val, &shape,
                        NULL, NULL, NULL, dtype_uint, dtype);  // work1 >other
        tpu_bdc_and(work2_addr, work0_addr, output_addr, &shape, NULL, NULL,
                    NULL, dtype_uint);  // work2 <0 && >other
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL,
                      dtype_uint);  // work0 true_val
        // work3 = work2 == work0 ? work1 : work3
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        // input==0 && other==0 -> output = 0
        tpu_bdc_equal_C(work0_addr, input_addr, positive_zero_C, true_val,
                        &shape, NULL, NULL, dtype_uint,
                        dtype);  // work0 input==0
        tpu_bdc_equal_C(work1_addr, other_addr, positive_zero_C, true_val,
                        &shape, NULL, NULL, dtype_uint,
                        dtype);  // work1 other==0
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL,
                    dtype_uint);  // work2 input==0 && other==0
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL,
                      dtype_uint);  // work0 true_val
        tpu_bdc_set_C(work1_addr, positive_zero_C, &shape, NULL,
                      dtype);  // work1 0
        // work3 = work2 == work0 ? work1 : output
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        // input==0 && other > 0 -> output = 0x00000001 or 0x0001
        tpu_bdc_equal_C(work0_addr, input_addr, positive_zero_C, true_val,
                        &shape, NULL, NULL, dtype_uint,
                        dtype);  // work0 input==0
        tpu_bdc_greater_C(work1_addr, other_addr, positive_zero_C, true_val,
                          &shape, NULL, NULL, dtype_uint,
                          dtype);  // work1 other>0
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL,
                    dtype_uint);  // work2 input==0 && other>0
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL,
                      dtype_uint);  // work0 true_val
        tpu_bdc_set_C(work1_addr, positive_C, &shape, NULL,
                      dtype_uint);  // work1 0x00000001 or 0x0001
        // output = work2 == work0 ? work1 : work3
        src3.context.addr = work3_addr;
        tpu_bdc_equal_select(output_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        // input==0 && other < 0 -> output = 0x80000001 or 0x8001
        tpu_bdc_equal_C(work0_addr, input_addr, positive_zero_C, true_val,
                        &shape, NULL, NULL, dtype_uint,
                        dtype);  // work0 input==0
        tpu_bdc_less_C(work1_addr, other_addr, positive_zero_C, true_val,
                       &shape, NULL, NULL, dtype_uint,
                       dtype);  // work1 other<0
        tpu_bdc_and(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL,
                    NULL,
                    dtype_uint);  // work2 input==0 && other<0
        tpu_bdc_set_C(work0_addr, true_val, &shape, NULL,
                      dtype_uint);  // work0 true_val
        tpu_bdc_set_C(work1_addr, negtive_C, &shape, NULL,
                      dtype_uint);  // work1 0x80000001 or 0x8001
        // work3 = work2 == work0 ? work1 : output
        src3.context.addr = output_addr;
        tpu_bdc_equal_select(work3_addr, &src0, &src1, &src2, &src3, &shape,
                             dtype_uint, dtype);

        isNan(work0_addr, input_addr, work1_addr, work2_addr, &shape, NULL,
              dtype);
        isNan(work1_addr, other_addr, work2_addr, output_addr, &shape, NULL,
              dtype);
        tpu_bdc_or(work2_addr, work0_addr, work1_addr, &shape, NULL, NULL, NULL,
                   dtype_uint);
        replaceWithNan(output_addr, work3_addr, work2_addr, work0_addr,
                       work1_addr, &shape, NULL, dtype);

        // Move out from local memory to global memory
        global_addr_t output_global_addr_gdma =
            output_global_addr +
            (ndone * output_global_stride.n + cdone * output_global_stride.c +
             hdone * output_global_stride.h) *
                tpu_data_type_size(dtype);
        tpu_gdma_cpy_L2S(output_global_addr_gdma, output_addr, &shape,
                         &output_global_stride, NULL, dtype);
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

void tpu_kernel_api_nextafter_bcast(const void *args) {
  sg_api_nextafter_bcast_t *api = (sg_api_nextafter_bcast_t *)args;
  TPUKERNEL_ASSERT(api->output_dim > 0 && api->output_dim <= 4);

  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

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
  nodechip_nextafter_bcast(api->input_global_addr, api->other_global_addr,
                           api->output_global_addr, &input_shape, &other_shape,
                           &output_shape, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafter_bcast);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_nextafter_bcast_multi_core(const void *args) {
  sg_api_nextafter_bcast_t *api = (sg_api_nextafter_bcast_t *)args;
  TPUKERNEL_ASSERT(api->output_dim > 0 && api->output_dim <= 4);

  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_BFP16);

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

  int dsize = tpu_data_type_size(api->dtype);

  int input_offset = (input_shape.n != 1 ? length_slice : 0) * core_idx *
                     input_shape.c * input_shape.h * input_shape.w * dsize;
  int other_offset = (other_shape.n != 1 ? length_slice : 0) * core_idx *
                     other_shape.c * other_shape.h * other_shape.w * dsize;
  int output_offset = (output_shape.n != 1 ? length_slice : 0) * core_idx *
                      output_shape.c * output_shape.h * output_shape.w * dsize;
  input_shape.n = input_shape.n != 1 ? cur_length_slice : 1;
  other_shape.n = other_shape.n != 1 ? cur_length_slice : 1;
  output_shape.n = MAX(input_shape.n, other_shape.n);

  if (core_idx * length_slice < length) {
    nodechip_nextafter_bcast(api->input_global_addr + input_offset,
                             api->other_global_addr + other_offset,
                             api->output_global_addr + output_offset,
                             &input_shape, &other_shape, &output_shape,
                             (data_type_t)api->dtype);
  }

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nextafter_bcast_multi_core);
#endif