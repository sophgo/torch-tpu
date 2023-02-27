#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

/**
 * Cross Entropy Loss (include Softmax):
 * 
 * forward:
 * input(batch, cls_num)
 * target(batch, cls_num)
 * 
 * 1. reduction_op = 0, reduction = "mean" (default)
 * loss = -(log(input.softmax(dim=1)) * target) / batch
 * 2. reduction_op = 1, reduction = "sum"
 * loss = log(input.softmax(dim=1)) * target
 * 
 */

static inline bool is_local_mem_enough(
    int batch,
    int cls,
    data_type_t dtype)
{
    int param_size = DIV_UP(batch, NPU_NUM) * tpu_aligned_feature_size(cls, 1, dtype);
    int table_size = tpu_aligned_feature_size(192, 1, dtype);
    int coeff_size = tpu_aligned_feature_size(32, 1, dtype);
    int exp_sum_size = DIV_UP(batch, NPU_NUM) * tpu_data_type_size(dtype);
    int pooling_addr = ALIGN(table_size + 2 * coeff_size + exp_sum_size, 64);
    int pooling_size = DIV_UP(batch, NPU_NUM) * 64;
    int total_size = ALIGN(param_size, BANK_SIZE) * 3 +
                     ALIGN(pooling_addr + pooling_size, BANK_SIZE);
    return total_size<LOCAL_MEM_SIZE;
}

void nodechip_cross_entropy_forward(
    global_addr_t input_global_addr,
    global_addr_t target_global_addr,
    global_addr_t loss_global_addr,
    int           batch,
    int           cls,
    int           reduction,
    data_type_t   dtype)
{
    TPUKERNEL_ASSERT(is_local_mem_enough(batch, cls, dtype));
    dim4 shape = {1, batch, 1, cls};
    dim4 nshape = {batch, 1, 1, 1};
    dim4 cshape = {1, batch, 1, 1};
    dim4 wshape = {1, 1, 1, batch};
    dim4 loss_shape = {1, 1, 1, 1};
    dim4 cdim_compact_stride;
    dim4 ndim_compact_stride;
    tpu_compact_stride(&cdim_compact_stride, 0, &cshape);
    tpu_compact_stride(&ndim_compact_stride, 0, &nshape);
    scalar_t scale = {.f32 = 1.};
    dim2 pooling_kernel = {1, cls};
    dim2 pooling_stride = {1, 1};
    dim2 pooling_dilation = {1, 1};
    padding_t pooling_padding = {0, 0, 0, 0};

    int param_size = DIV_UP(batch, NPU_NUM) * tpu_aligned_feature_size(cls, 1, dtype);
    int table_size = tpu_aligned_feature_size(192, 1, dtype);
    int coeff_size = tpu_aligned_feature_size(32, 1, dtype);
    int exp_sum_size = DIV_UP(batch, NPU_NUM) * tpu_data_type_size(dtype);
    int pooling_size = DIV_UP(batch, NPU_NUM) * 64;

    local_addr_t exp_table_local_addr = 0;
    local_addr_t exp_coeff_local_addr = exp_table_local_addr + table_size;
    local_addr_t log_coeff_local_addr = exp_coeff_local_addr + coeff_size;
    local_addr_t inv_expsum_local_addr = log_coeff_local_addr + coeff_size;
    local_addr_t exp_sum_local_addr = ALIGN(table_size + 2 * coeff_size + exp_sum_size, 64);
    local_addr_t input_local_addr = ALIGN(exp_sum_local_addr + pooling_size, BANK_SIZE);
    local_addr_t work0_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);
    local_addr_t work1_local_addr = work0_local_addr + ALIGN(param_size, BANK_SIZE);
    local_addr_t target_local_addr = work1_local_addr;

    // compute input.softmax(-1)
    tpu_parallel_start();
    tpu_bdc_load_fp32_exp_table(exp_table_local_addr);
    tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
    tpu_bdc_load_fp32_log_coeff(log_coeff_local_addr);
    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_parallel_end();
    tpu_bdc_fp32_exp(
        input_local_addr,
        input_local_addr,
        work0_local_addr,
        work1_local_addr,
        exp_coeff_local_addr,
        exp_table_local_addr,
        &shape);
    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        target_local_addr,
        target_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_avg_pool2d(
        exp_sum_local_addr,
        input_local_addr,
        &shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        DT_FP32,
        scale);
    tpu_bdc_fp32_C_div(
        inv_expsum_local_addr,
        exp_sum_local_addr,
        1.0f,
        &cshape,
        &cdim_compact_stride,
        NULL);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        inv_expsum_local_addr,
        &shape,
        dtype);
    // log(input.softmax(-1))*target
    tpu_bdc_fp32_log(
        input_local_addr,
        input_local_addr,
        work0_local_addr,
        log_coeff_local_addr,
        &shape);
    tpu_parallel_end();
    tpu_bdc_fp_mul(
        input_local_addr,
        input_local_addr,
        target_local_addr,
        &shape,
        NULL,
        NULL,
        NULL,
        dtype);
    // (1,b,1,cls) to (1,b,1,1)
    tpu_bdc_fp_avg_pool2d(
        exp_sum_local_addr,
        input_local_addr,
        &shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        DT_FP32,
        scale);
    // (1,b,1,1) to (1,1,1,b)
    tpu_gdma_cpy_nc_trans_L2L(
        0,
        exp_sum_local_addr,
        &nshape,
        &ndim_compact_stride,
        NULL,
        dtype);
    pooling_kernel.w = wshape.w;
    scale.f32 = reduction ? -1.f:-1.f / batch;
    // (1,1,1,b) to (1,1,1,1)
    tpu_bdc_fp_avg_pool2d(
        exp_sum_local_addr,
        0,
        &wshape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        DT_FP32,
        scale);
    tpu_gdma_cpy_L2S(
        loss_global_addr,
        exp_sum_local_addr,
        &loss_shape,
        NULL,
        NULL,
        dtype);
}

void tpu_kernel_api_cross_entropy_forward(const void *args)
{
    sg_api_crossentropy_forward_t *api = (sg_api_crossentropy_forward_t *)args;
    
    int reduction = api->reduction;
    TPUKERNEL_ASSERT(reduction==0||reduction==1||reduction==2);
    TPUKERNEL_ASSERT(!api->dtype);
    tpu_initialize();
    nodechip_cross_entropy_forward(
        api->input_global_addr,
        api->target_global_addr,
        api->loss_global_addr,
        api->batch,
        api->cls_num,
        api->reduction,
        DT_FP32);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_cross_entropy_forward)