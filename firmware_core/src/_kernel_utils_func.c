#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "kernel_utils_func.h"

void isNan(local_addr_t output_local_addr, local_addr_t input_local_addr,
                  local_addr_t work0_local_addr, local_addr_t work1_local_addr,
                  const dim4 *shape, const dim4 *stride, data_type_t dtype) {
  scalar_t exp_mask;
  scalar_t frac_mask;
  if (dtype == DT_FP32) {
    exp_mask.u32 = 0x7f800000;
    frac_mask.u32 = 0x007fffff;
  } else if (dtype == DT_FP16) {
    exp_mask.u16 = 0x7c00;
    frac_mask.u16 = 0x03ff;
  } else {
    exp_mask.u16 = 0x7f80;
    frac_mask.u16 = 0x007f;
  }
  scalar_t s_one = {.u8 = 1};
  scalar_t s_zero = {.u8 = 0};

  tpu_bdc_and_C(output_local_addr, input_local_addr, exp_mask, shape, stride, stride, dtype);
  tpu_bdc_equal_C(work0_local_addr, output_local_addr, exp_mask, s_one, shape, stride, stride, DT_UINT8,
                  dtype);

  tpu_bdc_and_C(output_local_addr, input_local_addr, frac_mask, shape, stride,
                stride, dtype);
  tpu_bdc_not_equal_C(work1_local_addr, output_local_addr, s_zero, s_one, shape,
                      stride, stride, DT_UINT8, dtype);

  tpu_bdc_and(output_local_addr, work1_local_addr, work0_local_addr, shape,
              stride, stride, stride, DT_UINT8);
}

void replaceWithNan(local_addr_t output_local_addr,
                           local_addr_t input_local_addr,
                           local_addr_t mask_local_addr,
                           local_addr_t work0_local_addr,
                           local_addr_t work1_local_addr, const dim4 *shape,
                           const dim4 *stride, data_type_t dtype) {
  scalar_t C = {.u8 = 1};
  scalar_t nan_C;
  if (dtype == DT_FP32) {
    nan_C.u32 = 0x7f800001;
  } else if (dtype == DT_FP16) {
    nan_C.u16 = 0x7c01;
  } else {
    nan_C.u16 = 0x7f81;
  }
  tpu_bdc_set_C(work0_local_addr, C, shape, stride, DT_UINT8);
  tpu_bdc_set_C(work1_local_addr, nan_C, shape, stride, dtype);
  variable_t src0 = {.type = TENSOR, .context = {.addr = mask_local_addr}};
  variable_t src1 = {.type = TENSOR, .context = {.addr = work0_local_addr}};
  variable_t src2 = {.type = TENSOR, .context = {.addr = work1_local_addr}};
  variable_t src3 = {.type = TENSOR, .context = {.addr = input_local_addr}};
  tpu_bdc_equal_select(output_local_addr, &src0, &src1, &src2, &src3, shape,
                       DT_UINT8, dtype);
}

void tpu_bdc_fp_isinf(local_addr_t dst_addr, local_addr_t src_addr,
                      local_addr_t work0_addr, const dim4 *shape,
                      data_type_t dtype) {
  scalar_t inf_C = {.u32 = (dtype == DT_FP32
                                ? 0x7f800000
                                : (dtype == DT_FP16 ? 0x7c00 : 0x7f80))};
  scalar_t neg_C = {.u32 = (dtype == DT_FP32
                                ? 0x7fffffff
                                : (dtype == DT_FP16 ? 0x7fff : 0x7fff))};
  scalar_t C = {.u8 = 1};

  tpu_bdc_and_C(work0_addr, src_addr, neg_C, shape, NULL, NULL, dtype);
  tpu_bdc_equal_C(dst_addr, work0_addr, inf_C, C, shape, NULL, NULL, DT_UINT8,
                  dtype);
}

void tpu_bdc_fp_isnan(local_addr_t dst_addr, local_addr_t src_addr,
                      local_addr_t work0_addr, local_addr_t work1_addr,
                      local_addr_t work2_addr, const dim4 *shape,
                      data_type_t dtype) {
  scalar_t exp_mask = {.u32 = (dtype == DT_FP32
                                ? 0x7f800000
                                : (dtype == DT_FP16 ? 0x7c00 : 0x7f80))};
  scalar_t frac_mask = {.u32 = (dtype == DT_FP32
                                ? 0x007fffff
                                : (dtype == DT_FP16 ? 0x03ff : 0x007f))};
  scalar_t s_one = {.u8 = 1};
  scalar_t s_zero = {.u8 = 0};

  tpu_bdc_and_C(work2_addr, src_addr, exp_mask, shape, NULL, NULL, dtype);
  tpu_bdc_equal_C(work0_addr, work2_addr, exp_mask, s_one, shape, NULL, NULL, DT_UINT8,
                  dtype);

  tpu_bdc_and_C(work2_addr, src_addr, frac_mask, shape, NULL, NULL, dtype);
  tpu_bdc_not_equal_C(work1_addr, work2_addr, s_zero, s_one, shape, NULL, NULL,
                      DT_UINT8, dtype);

  tpu_bdc_and(dst_addr, work0_addr, work1_addr, shape, NULL, NULL, NULL,
              DT_UINT8);
}