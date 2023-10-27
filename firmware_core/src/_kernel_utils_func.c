#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "kernel_utils_func.h"

void isNan(local_addr_t output_local_addr, local_addr_t input_local_addr,
                  local_addr_t work0_local_addr, local_addr_t work1_local_addr,
                  const dim4 *shape, const dim4 *stride, data_type_t dtype) {
  scalar_t inf_C;
  scalar_t neg_C;
  if (dtype == DT_FP32) {
    inf_C.u32 = 0x7f800000;
    neg_C.u32 = 0x7fffffff;
  } else if (dtype == DT_FP16) {
    inf_C.u16 = 0x7c00;
    neg_C.u16 = 0x7fff;
  } else {
    inf_C.u16 = 0x7f80;
    neg_C.u16 = 0x7fff;
  }
  scalar_t C = {.u8 = 1};
  tpu_bdc_and_C(work1_local_addr, input_local_addr, inf_C, shape, stride,
                stride, dtype);
  tpu_bdc_equal_C(output_local_addr, work1_local_addr, inf_C, C, shape, stride,
                  stride, DT_UINT8, dtype);
  tpu_bdc_and_C(work1_local_addr, input_local_addr, neg_C, shape, stride,
                stride, dtype);
  tpu_bdc_not_equal_C(work0_local_addr, work1_local_addr, inf_C, C, shape,
                      stride, stride, DT_UINT8, dtype);
  tpu_bdc_and(work1_local_addr, output_local_addr, work0_local_addr, shape,
              stride, stride, stride, DT_UINT8);
  tpu_bdc_equal_C(output_local_addr, work1_local_addr, C, C, shape, stride,
                  stride, DT_UINT8, DT_UINT8);
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
  scalar_t inf_C = {.u32 = (dtype == DT_FP32
                                ? 0x7f800000
                                : (dtype == DT_FP16 ? 0x7c00 : 0x7f80))};
  scalar_t neg_C = {.u32 = (dtype == DT_FP32
                                ? 0x7fffffff
                                : (dtype == DT_FP16 ? 0x7fff : 0x7fff))};
  scalar_t C = {.u8 = 1};

  tpu_bdc_and_C(work2_addr, src_addr, inf_C, shape, NULL, NULL, dtype);
  tpu_bdc_equal_C(work0_addr, work2_addr, inf_C, C, shape, NULL, NULL, DT_UINT8,
                  dtype);

  tpu_bdc_and_C(work2_addr, src_addr, neg_C, shape, NULL, NULL, dtype);
  tpu_bdc_not_equal_C(work1_addr, work2_addr, inf_C, C, shape, NULL, NULL,
                      DT_UINT8, dtype);

  tpu_bdc_and(work2_addr, work0_addr, work1_addr, shape, NULL, NULL, NULL,
              DT_UINT8);
  tpu_bdc_equal_C(dst_addr, work2_addr, C, C, shape, NULL, NULL, DT_UINT8,
                  DT_UINT8);
}