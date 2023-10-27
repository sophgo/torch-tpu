#ifndef TPU_TRAIN_KERNEL_UTILS_FUNC_H
#define TPU_TRAIN_KERNEL_UTILS_FUNC_H
#include "sg_api_struct.h"
#include "tpu_kernel.h"

void isNan(
    local_addr_t output_local_addr, local_addr_t input_local_addr,
    local_addr_t work0_local_addr, local_addr_t work1_local_addr,
    const dim4 *shape, const dim4 *stride, data_type_t dtype );

void replaceWithNan(
    local_addr_t output_local_addr,
    local_addr_t input_local_addr,
    local_addr_t mask_local_addr,
    local_addr_t work0_local_addr,
    local_addr_t work1_local_addr,
    const dim4 *shape,
    const dim4 *stride,
    data_type_t dtype
);

void tpu_bdc_fp_isinf(local_addr_t dst_addr, local_addr_t src_addr,
                      local_addr_t work0_addr, const dim4 *shape,
                      data_type_t dtype);

void tpu_bdc_fp_isnan(local_addr_t dst_addr, local_addr_t src_addr,
                      local_addr_t work0_addr, local_addr_t work1_addr,
                      local_addr_t work2_addr, const dim4 *shape,
                      data_type_t dtype);

#endif