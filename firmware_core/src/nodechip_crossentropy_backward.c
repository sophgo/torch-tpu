#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
/**
 * Cross Etropy Loss (include Softmax):
 * 
 * backward:
 * input(batch, cls_num)
 * target(batch, cls_num)
 * 
 * 1. reduction_op = 0, reduction = "mean" (default)
 * input.grad = (input.softmax(dim=1)-target)/batch
 * 2. reduction_op = 1, reduction = "sum"
 * input.grad = input.softmax(dim=1)-target
 */

static inline bool is_local_mem_enough(
    int batch,
    int cls,
    data_type_t dtype)
{
    int table_size = tpu_aligned_feature_size(192, 1, DT_FP32);
    int coeff_size = tpu_aligned_feature_size(32, 1, DT_FP32);
    int exp_sum_size = ALIGN(DIV_UP(batch, NPU_NUM) * tpu_data_type_size(DT_FP32), 4);
    int param_size = ALIGN(DIV_UP(batch, NPU_NUM) * tpu_aligned_feature_size(cls, 1, dtype), BANK_SIZE);
    int total_size = param_size * 3 + ALIGN(table_size + coeff_size + exp_sum_size * 2, BANK_SIZE);
    return total_size<LOCAL_MEM_SIZE;
}

void nodechip_cross_entropy_backward(
    global_addr_t input_global_addr,
    global_addr_t target_global_addr,
    global_addr_t grad_input_global_addr,
    int           batch,
    int           cls,
    int           reduction,
    data_type_t   dtype)
{
    TPUKERNEL_ASSERT(is_local_mem_enough(batch,cls,dtype));
    dim4 shape = {
        .n = 1,
        .c = batch,
        .h = 1,
        .w = cls};
    dim4 cshape = {
        .n = 1,
        .c = batch,
        .h = 1,
        .w = 1};
    dim4 compact_stride;
    tpu_compact_stride(&compact_stride,0,&cshape);
    scalar_t scale = {.f32 = 1.};
    dim2 pooling_kernel = {1, cls};
    dim2 pooling_stride = {1, 1};
    dim2 pooling_dilation = {1, 1};
    padding_t pooling_padding = {0, 0, 0, 0};

    int param_size = DIV_UP(batch, NPU_NUM) * tpu_aligned_feature_size(cls, 1, dtype);
    int table_size = tpu_aligned_feature_size(192, 1, dtype);
    int coeff_size = tpu_aligned_feature_size(32, 1, dtype);
    int exp_sum_size = DIV_UP(batch, NPU_NUM) * tpu_data_type_size(dtype);

    local_addr_t table_local_addr = 0;
    local_addr_t coeff_local_addr = table_local_addr + table_size;
    local_addr_t exp_sum_local_addr = coeff_local_addr + coeff_size;
    local_addr_t inv_expsum_local_addr = exp_sum_local_addr + exp_sum_size;
    local_addr_t input_local_addr = ALIGN(table_size + coeff_size + 2 * exp_sum_size, BANK_SIZE);
    local_addr_t work0_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);
    local_addr_t work1_local_addr = work0_local_addr + ALIGN(param_size, BANK_SIZE);
    local_addr_t target_local_addr = work0_local_addr;
    local_addr_t grad_input_local_addr = work1_local_addr;

    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_load_fp32_exp_coeff(coeff_local_addr);
    tpu_bdc_load_fp32_exp_table(table_local_addr);
    tpu_parallel_end();
    tpu_bdc_fp32_exp(
        input_local_addr,
        input_local_addr,
        work0_local_addr,
        work1_local_addr,
        coeff_local_addr,
        table_local_addr,
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
        &compact_stride,
        &compact_stride);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        inv_expsum_local_addr,
        &shape,
        dtype);
    tpu_parallel_end();
    tpu_bdc_fp_sub(
        grad_input_local_addr,
        input_local_addr,
        target_local_addr,
        &shape,
        NULL,
        NULL,
        NULL,
        dtype);
    if(!reduction)
    {
        scale.f32 = 1/batch;
        tpu_bdc_fp_mul_C(
            grad_input_local_addr,
            grad_input_local_addr,
            scale,
            &shape,
            NULL,
            NULL,
            dtype);
    }
    tpu_gdma_cpy_L2S(
        grad_input_global_addr,
        grad_input_local_addr,
        &shape,
        NULL,
        NULL,
        dtype);
}

void tpu_kernel_api_cross_entropy_backward(const void *args)
{
    sg_api_crossentropy_backward_t *api = (sg_api_crossentropy_backward_t *)args;
    
    int reduction = api->reduction;
    TPUKERNEL_ASSERT(reduction==0||reduction==1||reduction==2);
    TPUKERNEL_ASSERT(!api->dtype);
    tpu_initialize();
    nodechip_cross_entropy_backward(
        api->input_global_addr,
        api->target_global_addr,
        api->grad_input_global_addr,
        api->batch,
        api->cls_num,
        reduction,
        tpu_type_convert(api->dtype));
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_cross_entropy_backward);