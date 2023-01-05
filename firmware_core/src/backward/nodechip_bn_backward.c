#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#ifdef USING_CMODEL
#include "cmodel_memory.h"
#endif

typedef struct{
    unsigned long long grad_output_global_addr;
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long saved_mean_global_addr;
    unsigned long long saved_invstd_global_addr;
    unsigned long long grad_input_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long grad_bias_global_addr;
    int                shape[4];
    int                grad_input_enable;
    int                grad_weight_enable;
    int                grad_bias_enable;
} __attribute__((packed)) sg_api_bn_backward_t;

/**
 * train mode：n = batch, track_running_stat = true
 * forward：
 * 1. compute batch_mean and batch_var
 * 2. compute bn result = (x - batch_mean) / sqrt(batch_var + eps) * weight + bias
 * 3. update running_mean and running_var = momentum * running_stat + (1-momentum) * batch_stat
 * 4. save batch_mean, batch_invStd = rsqrt(var + eps) for backward
 * 
 * backward:
 *  -------------------------------------------------------------------------------
 *  |        cut dim-c        |     cut dim-hw loop1    |    cut dim-hw loop2     |
 *  -------------------------------------------------------------------------------
 *  | param = 3 * (n,c,h,w)   | param = 2 * (n,c,h,w)   | param = 2 * (n,c,h,w)   |
 *  | c_param = 3 * (1,c,1,1) | c_param = 5 * (1,c,1,1) | c_param = 5 * (1,c,1,1) |
 *  | buffer1 = (1,c,1,1)     | buffer1 = (1,c,1,1)     |                         |
 *  | buffer2 = (1,c,n,1)     | buffer2 = (1,c,n,1)     |                         |
 *  | buffer3 = (1,c,n,hw)    | buffer3 = (1,c,n,hw)    |                         |
 * -------------------------------------------------------------------------------
*/

static inline bool is_local_mem_enough(
    int n,
    int c,
    int hw,
    int mode,
    data_type_t dtype)
{
    int c_param_size = ALIGN(DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int param_size = n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(hw, 1, dtype);
    int buffer_size1 = DIV_UP(c,NPU_NUM) * 64;
    int buffer_size2 = n == 1 ? 0 : DIV_UP(c,NPU_NUM) * tpu_aligned_feature_size(n, 1, DT_FP32);
    int buffer_size3 = dtype == DT_FP32 ? 0 : n * DIV_UP(c,NPU_NUM) * tpu_aligned_feature_size(hw, 1, DT_FP32);
    int buffer_size = buffer_size1 + buffer_size2 + buffer_size3;
    // mode 0 : split c
    // mode 1 : split hw first loop
    // mode 2 : split hw second loop
    int total_size = mode==0 ? ALIGN(c_param_size * 3, BANK_SIZE) +
                               ALIGN(param_size, BANK_SIZE) * 3 +
                               ALIGN(buffer_size, BANK_SIZE):
                     mode==1 ? ALIGN(c_param_size * 4, BANK_SIZE) +
                               ALIGN(param_size, BANK_SIZE) * 2 +
                               ALIGN(buffer_size, BANK_SIZE):
                               ALIGN(c_param_size * 4, BANK_SIZE) +
                               ALIGN(param_size, BANK_SIZE) * 2;
    return total_size < LOCAL_MEM_SIZE;
}

void pooling_fp16_to_fp32(
    local_addr_t pooling_addr,
    local_addr_t pooling_fp32,
    local_addr_t pooling_buffer1,
    local_addr_t pooling_buffer2,
    dim4 input_shape,
    dim4 pooling_shape,
    data_type_t dtype)
{
    dim4 cshape = {
        .n = 1,
        .c = input_shape.c,
        .h = 1,
        .w = 1};
    dim4 compact_stride;
    dim4 aligned_stride;
    dim4 local_stride;
    tpu_compact_stride(&compact_stride, 0, &cshape);
    tpu_aligned_stride(&aligned_stride, 0, &input_shape, dtype);
    tpu_aligned_stride(&local_stride, 0, &input_shape, DT_FP32);
    if(input_shape.c > NPU_NUM)
    {
        local_stride.n = local_stride.c;
        local_stride.c *= input_shape.n;
    }
    scalar_t scale = {.f32 = 1.};
    dim2 pooling_kernel = {1, pooling_shape.w};
    dim2 pooling_stride = {1, 1};
    dim2 pooling_dilation = {1, 1};
    padding_t pooling_padding = {0, 0, 0, 0};
    if(dtype!=DT_FP32)
    {
        tpu_bdc_cast(
            pooling_fp32,
            pooling_addr,
            &input_shape,
            &local_stride,
            &aligned_stride,
            DT_FP32,
            dtype,
            0);
    }
    // {1,c,n,hw} to {1,c,n,1}
    tpu_bdc_fp_avg_pool2d(
        pooling_buffer1,
        dtype == DT_FP32 ? pooling_addr : pooling_fp32,
        &pooling_shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        DT_FP32,
        scale);
    if(input_shape.n>1)
    {
        pooling_shape.w = 1;
        pooling_kernel.w = 1;
        pooling_kernel.h = pooling_shape.h;
        // {1,c,n,1} to {1,c,1,1}
        tpu_bdc_fp_avg_pool2d(
            pooling_buffer2,
            pooling_buffer1,
            &pooling_shape,
            &pooling_kernel,
            &pooling_padding,
            &pooling_stride,
            &pooling_dilation,
            DT_FP32,
            scale);
    }
}

void bn_backward_split_c(
    global_addr_t grad_output_global_addr,
    global_addr_t input_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t mean_global_addr,
    global_addr_t invstd_global_addr,
    global_addr_t grad_input_global_addr,
    global_addr_t grad_weight_global_addr,
    global_addr_t grad_bias_global_addr,
    dim4 ori_shape,
    int csecs,
    data_type_t dtype,
    int grad_weight_enable,
    int grad_bias_enable)
{   
    int n = ori_shape.n;
    int h = ori_shape.h;
    int w = ori_shape.w;
    dim4 input_shape = {
        .n = n,
        .c = csecs,
        .h = h,
        .w = w};
    dim4 cshape = {
        .n = 1,
        .c = csecs,
        .h = 1,
        .w = 1};
    dim4 pooling_shape = {
        .n = 1,
        .c = csecs,
        .h = n,
        .w = ALIGN(h * w, tpu_eu_num(DT_FP32))};
    int c_param_size = ALIGN(DIV_UP(csecs, NPU_NUM) * tpu_data_type_size(dtype),4);
    int param_size = n * DIV_UP(csecs, NPU_NUM) * tpu_aligned_feature_size(h, w, dtype);
    int buffer_size1 = DIV_UP(csecs, NPU_NUM) * 64; 
    int buffer_size2 = n == 1 ? 0 : DIV_UP(csecs, NPU_NUM) * tpu_aligned_feature_size(n, 1, DT_FP32);
    int buffer_size3 = dtype == DT_FP32 ? 0 : n * DIV_UP(csecs, NPU_NUM) * tpu_aligned_feature_size(h, w, DT_FP32);
    
    //param (1,c,1,1) compact
    local_addr_t weight_local_addr = 0;
    local_addr_t invstd_local_addr = weight_local_addr + c_param_size;
    local_addr_t mean_local_addr = invstd_local_addr + c_param_size;
    local_addr_t grad_bias_local_addr = invstd_local_addr;
    local_addr_t grad_weight_local_addr = mean_local_addr;
    //param (1,c,1,1)+(1,c,n,1)+(1,c,n,hw) aligned
    local_addr_t pooling_buffer1 = ALIGN(c_param_size * 3, BANK_SIZE);
    local_addr_t pooling_buffer2 = pooling_buffer1 + buffer_size2;
    local_addr_t pooling_fp32 = pooling_buffer1 + buffer_size1 + buffer_size2;
    //param (n,c,h,w) aligned
    local_addr_t input_local_addr = pooling_buffer1 + ALIGN(buffer_size1 + buffer_size2 + buffer_size3, BANK_SIZE);
    local_addr_t grad_output_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);
    local_addr_t prod_local_addr = grad_output_local_addr + ALIGN(param_size, BANK_SIZE);

    dim4 compact_stride;
    dim4 aligned_stride;
    dim4 local_stride;
    dim4 global_stride;
    tpu_compact_stride(&compact_stride, 0, &cshape);
    tpu_aligned_stride(&aligned_stride, 0, &input_shape, dtype);
    tpu_aligned_stride(&local_stride, 0, &input_shape, DT_FP32);
    tpu_continuous_stride(&global_stride ,&ori_shape);
    scalar_t scale = {.f32 = 0};  
    if(csecs > NPU_NUM)
    {
        local_stride.n = local_stride.c;
        local_stride.c *= n;
    }
    if((h * w) % (tpu_eu_num(DT_FP32)))
    {
        tpu_parallel_start();
        if(dtype==DT_FP32)
        {
            tpu_bdc_set_C(
                grad_output_local_addr,
                scale,
                &pooling_shape,
                NULL,
                DT_FP32);
            tpu_bdc_set_C(
                prod_local_addr,
                scale,
                &pooling_shape,
                NULL,
                DT_FP32);
        }
        else
        {
            tpu_bdc_set_C(
                pooling_fp32,
                scale,
                &pooling_shape,
                NULL,
                DT_FP32);
        }
    }
    tpu_gdma_compact_S2L(
        weight_local_addr,
        weight_global_addr,
        &cshape,
        dtype);
    tpu_gdma_compact_S2L(
        invstd_local_addr,
        invstd_global_addr,
        &cshape,
        dtype);
    tpu_gdma_compact_S2L(
        mean_local_addr,
        mean_global_addr,
        &cshape,
        dtype);
    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &input_shape,
        &aligned_stride,
        &global_stride,
        dtype);
    if(tpu_is_parallel_state()){tpu_parallel_end();}
    // compute x_norm
    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &input_shape,
        dtype==DT_FP32? &local_stride : &aligned_stride,
        &global_stride,
        dtype);
    tpu_bdc_fp_mul(
        weight_local_addr,
        weight_local_addr,
        invstd_local_addr,
        &cshape,
        &compact_stride,
        &compact_stride,
        &compact_stride,
        dtype); 
    tpu_bdc_neg(
        mean_local_addr,
        mean_local_addr,
        &cshape,
        &compact_stride,
        &compact_stride,
        dtype);
    tpu_bdc_fp_bias(
        input_local_addr,
        input_local_addr,
        mean_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        invstd_local_addr,
        &input_shape,
        dtype);
    tpu_parallel_end();
    // compute grad_bias
    pooling_fp16_to_fp32(
        grad_output_local_addr,
        pooling_fp32,
        pooling_buffer1,
        pooling_buffer2,
        input_shape,
        pooling_shape,
        dtype);
    if(dtype!=DT_FP32)
    {
        tpu_bdc_cast(
            grad_bias_local_addr,
            pooling_buffer2,
            &cshape,
            &compact_stride,
            NULL,
            dtype,
            DT_FP32,
            RM_HALF_AWAY_FROM_ZERO);
    }
    else
    {
        tpu_bdc_cpy(
            grad_bias_local_addr,
            pooling_buffer2,
            &cshape,
            &compact_stride,
            NULL,
            dtype);
    }
    // compute grad_weight
    if(grad_bias_enable)
    {
        tpu_parallel_start();
        tpu_gdma_compact_L2S(
            grad_bias_global_addr,
            grad_bias_local_addr,
            &cshape,
            dtype);
    }
    tpu_bdc_fp_mul(
        prod_local_addr,
        grad_output_local_addr,
        input_local_addr,
        &input_shape,
        dtype==DT_FP32? &local_stride : &aligned_stride,
        dtype==DT_FP32? &local_stride : &aligned_stride,
        &aligned_stride,
        dtype);
    pooling_fp16_to_fp32(
        prod_local_addr,
        pooling_fp32,
        pooling_buffer1,
        pooling_buffer2,
        input_shape,
        pooling_shape,
        dtype);
    if(dtype!=DT_FP32)
    {
        tpu_bdc_cast(
            grad_weight_local_addr,
            pooling_buffer2,
            &cshape,
            &compact_stride,
            NULL,
            dtype,
            DT_FP32,
            RM_HALF_AWAY_FROM_ZERO);
    }
    else
    {
        tpu_bdc_cpy(
            grad_weight_local_addr,
            pooling_buffer2,
            &cshape,
            &compact_stride,
            NULL,
            dtype);
    }
    if(tpu_is_parallel_state()) tpu_parallel_end();
    //compute grad_out
    if(grad_weight_enable)
    {
        tpu_parallel_start();
        tpu_gdma_compact_L2S(
            grad_weight_global_addr,
            grad_weight_local_addr,
            &cshape,
            dtype);
    }
    // part1: (γ/σ)*dL/dy
    if(dtype!=DT_FP32)
    {
        tpu_bdc_fp_scale(
            grad_output_local_addr,
            grad_output_local_addr,
            weight_local_addr,
            &input_shape,
            dtype);
    }
    else
    {
        dim4 zero_stride = {
            .n = 0,
            .c = 1,
            .h = 0,
            .w = 0};
        tpu_bdc_fp_mul(
            grad_output_local_addr,
            grad_output_local_addr,
            weight_local_addr,
            &input_shape,
            &local_stride,
            &local_stride,
            &zero_stride,
            dtype);
    }
    // part2: (dL/dγ * x_norm + dL/dβ) * γ/σ/N
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        grad_weight_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_bias(
        input_local_addr,
        input_local_addr,
        grad_bias_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        weight_local_addr,
        &input_shape,
        dtype);
    scale.f32 = 1./(n*h*w);
    tpu_bdc_fp_mul_C(
        input_local_addr,
        input_local_addr,
        tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        &input_shape,
        &aligned_stride,
        &aligned_stride,
        dtype);
    tpu_bdc_fp_sub(
        grad_output_local_addr,
        grad_output_local_addr,
        input_local_addr,
        &input_shape,
        dtype==DT_FP32? &local_stride : &aligned_stride,
        dtype==DT_FP32? &local_stride : &aligned_stride,
        &aligned_stride,
        dtype);
    if(tpu_is_parallel_state()) tpu_parallel_end();
    tpu_gdma_cpy_L2S(
        grad_input_global_addr,
        grad_output_local_addr,
        &input_shape,
        &global_stride,
        dtype==DT_FP32? &local_stride : &aligned_stride,
        dtype);
}

void accumulate_grad_weight_bias(
    global_addr_t grad_output_global_addr,
    global_addr_t input_global_addr,
    global_addr_t mean_global_addr,
    global_addr_t invstd_global_addr,
    global_addr_t weight_global_addr,
    dim4 ori_shape,
    int hwsecs,
    int hwidx,
    data_type_t dtype)
{
    TPUKERNEL_ASSERT(!(hwsecs%tpu_eu_num(dtype)));
    int n = ori_shape.n;
    int c = ori_shape.c;
    dim4 input_shape = {
        .n = n,
        .c = c,
        .h = 1,
        .w = hwsecs};
    dim4 cshape = {
        .n = 1,
        .c = c,
        .h = 1,
        .w = 1};
    dim4 pooling_shape = {
        .n = 1,
        .c = c,
        .h = n,
        .w = hwsecs};
    int c_param_size = ALIGN(DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype),4);
    int param_size = n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(hwsecs, 1, dtype);
    int buffer_size1 = DIV_UP(c, NPU_NUM) * 64; 
    int buffer_size2 = n == 1 ? 0 : DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(n, 1, dtype);
    int buffer_size3 = dtype == DT_FP32 ? 0 : n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(hwsecs, 1, DT_FP32);

    //param (1,c,1,1) compact
    local_addr_t grad_bias_local_addr = 0;
    local_addr_t grad_weight_local_addr = grad_bias_local_addr + c_param_size;
    local_addr_t invstd_local_addr = grad_weight_local_addr + c_param_size;
    local_addr_t mean_local_addr = invstd_local_addr + c_param_size;
    local_addr_t weight_local_addr = mean_local_addr + c_param_size;
    //param (1,c,n,1) aligned
    local_addr_t pooling_buffer1 = ALIGN(c_param_size * 5, BANK_SIZE);
    local_addr_t pooling_buffer2 = pooling_buffer1 + buffer_size2;
    local_addr_t pooling_fp32 = pooling_buffer1 + buffer_size1 + buffer_size2;
    //param (n,c,h,w) aligned
    local_addr_t input_local_addr = pooling_buffer1 + ALIGN(buffer_size1 + buffer_size2 + buffer_size3, BANK_SIZE);
    local_addr_t grad_output_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);

    dim4 compact_stride;
    dim4 aligned_stride;
    dim4 local_stride;
    dim4 global_stride;
    tpu_compact_stride(&compact_stride, 0, &cshape);
    tpu_aligned_stride(&aligned_stride, 0, &input_shape, dtype);
    tpu_aligned_stride(&local_stride, 0, &input_shape, DT_FP32);
    tpu_continuous_stride(&global_stride ,&ori_shape);
    scalar_t scale = {.f32 = 0};
    if(c > NPU_NUM)
    {
        local_stride.n = local_stride.c;
        local_stride.c *= n;
    }  
    if(!hwidx)
    {
        tpu_parallel_start();
        tpu_bdc_set_C(
            grad_bias_local_addr,
            tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            &cshape,
            &compact_stride,
            dtype);
        tpu_bdc_set_C(
            grad_weight_local_addr,
            tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            &cshape,
            &compact_stride,
            dtype);
        tpu_gdma_compact_S2L(
            invstd_local_addr,
            invstd_global_addr,
            &cshape,
            dtype);
        tpu_gdma_compact_S2L(
            mean_local_addr,
            mean_global_addr,
            &cshape,
            dtype);
        tpu_gdma_compact_S2L(
            weight_local_addr,
            weight_global_addr,
            &cshape,
            dtype);
        tpu_parallel_end();
        tpu_bdc_fp_mul(
            weight_local_addr,
            weight_local_addr,
            invstd_local_addr,
            &cshape,
            &compact_stride,
            &compact_stride,
            &compact_stride,
            dtype);
        tpu_bdc_neg(
            mean_local_addr,
            mean_local_addr,
            &cshape,
            &compact_stride,
            &compact_stride,
            dtype);
    }

    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &input_shape,
        dtype==DT_FP32 ? &local_stride : &aligned_stride,
        &global_stride,
        dtype);
    // compute x_norm_part
    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &input_shape,
        dtype==DT_FP32 ? &local_stride : &aligned_stride,
        &global_stride,
        dtype);
    tpu_bdc_fp_bias(
        input_local_addr,
        input_local_addr,
        mean_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        invstd_local_addr,
        &input_shape,
        dtype);
    tpu_parallel_end();
    // compute pooling_part
    tpu_bdc_fp_mul(
        input_local_addr,
        input_local_addr,
        grad_output_local_addr,
        &input_shape,
        dtype==DT_FP32 ? &local_stride : &aligned_stride,
        dtype==DT_FP32 ? &local_stride : &aligned_stride,
        dtype==DT_FP32 ? &local_stride : &aligned_stride,
        dtype);
    pooling_fp16_to_fp32(
        grad_output_local_addr,
        pooling_fp32,
        pooling_buffer1,
        pooling_buffer2,
        input_shape,
        pooling_shape,
        dtype);
    if(dtype!=DT_FP32)
    {
        tpu_bdc_cast(
            pooling_fp32,
            pooling_buffer2,
            &cshape,
            &compact_stride,
            NULL,
            dtype,
            DT_FP32,
            RM_HALF_AWAY_FROM_ZERO);
    }
    tpu_bdc_fp_add(
        grad_bias_local_addr,
        grad_bias_local_addr,
        dtype == DT_FP32 ? pooling_buffer2 : pooling_fp32,
        &cshape,
        &compact_stride,
        &compact_stride,
        &compact_stride,
        dtype);
    pooling_fp16_to_fp32(
        input_local_addr,
        pooling_fp32,
        pooling_buffer1,
        pooling_buffer2,
        input_shape,
        pooling_shape,
        dtype);
    if(dtype!=DT_FP32)
    {
        tpu_bdc_cast(
            pooling_fp32,
            pooling_buffer2,
            &cshape,
            &compact_stride,
            NULL,
            dtype,
            DT_FP32,
            RM_HALF_AWAY_FROM_ZERO);
    }
    tpu_bdc_fp_add(
        grad_weight_local_addr,
        grad_weight_local_addr,
        dtype == DT_FP32 ? pooling_buffer2 : pooling_fp32,
        &cshape,
        &compact_stride,
        &compact_stride,
        &compact_stride,
        dtype);
}

void bn_backward_split_hw(
    global_addr_t grad_output_global_addr,
    global_addr_t input_global_addr,
    global_addr_t grad_input_global_addr,
    global_addr_t grad_weight_global_addr,
    global_addr_t grad_bias_global_addr,
    dim4 ori_shape,
    int hwsecs,
    int hwidx,
    data_type_t dtype,
    int grad_weight_enable,
    int grad_bias_enable)
{
    int n = ori_shape.n;
    int c = ori_shape.c;
    int h = ori_shape.h;
    int w = ori_shape.w;
    dim4 input_shape = {
        .n = n,
        .c = c,
        .h = 1,
        .w = hwsecs};
    dim4 cshape = {
        .n = 1,
        .c = c,
        .h = 1,
        .w = 1};
    int c_param_size = ALIGN(DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int param_size = n * DIV_UP(c,NPU_NUM) * tpu_aligned_feature_size(hwsecs, 1, dtype);
    //param (1,c,1,1) compact
    local_addr_t grad_bias_local_addr = 0;
    local_addr_t grad_weight_local_addr = grad_bias_local_addr + c_param_size;
    local_addr_t invstd_local_addr = grad_weight_local_addr + c_param_size;
    local_addr_t mean_local_addr = invstd_local_addr + c_param_size;
    local_addr_t weight_local_addr = mean_local_addr + c_param_size;
    //param (n,c,h,w) aligned
    local_addr_t input_local_addr = ALIGN(c_param_size * 5, BANK_SIZE);
    local_addr_t grad_output_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);

    dim4 aligned_stride;
    dim4 global_stride;
    dim4 compact_stride;
    tpu_aligned_stride(&aligned_stride, 0, &input_shape, dtype);
    tpu_continuous_stride(&global_stride, &ori_shape);
    tpu_compact_stride(&compact_stride, 0, &cshape);
    
    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &input_shape,
        &aligned_stride,
        &global_stride,
        dtype);
    if(!hwidx)
    {
        if(grad_bias_enable)
        {
            tpu_parallel_start();
            tpu_gdma_compact_L2S(
                grad_bias_global_addr,
                grad_bias_local_addr,
                &cshape,
                dtype);
        }
        if(grad_weight_enable)
        {
            if(!tpu_is_parallel_state())tpu_parallel_start();
            tpu_gdma_compact_L2S(
                grad_weight_global_addr,
                grad_weight_local_addr,
                &cshape,
                dtype);
        }
    }
    if(!tpu_is_parallel_state())tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &input_shape,
        &aligned_stride,
        &global_stride,
        dtype);
    tpu_bdc_fp_bias(
        input_local_addr,
        input_local_addr,
        mean_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        invstd_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        grad_weight_local_addr,
        &input_shape,
        dtype);
    tpu_bdc_fp_bias(
        input_local_addr,
        input_local_addr,
        grad_bias_local_addr,
        &input_shape,
        dtype);
    scalar_t scale = {.f32 = 1./(n * h * w)};
    tpu_bdc_fp_mul_C(
        input_local_addr,
        input_local_addr,
        tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        &input_shape,
        &aligned_stride,
        &aligned_stride,
        dtype);
    tpu_parallel_end();    
    tpu_bdc_fp_sub(
        grad_output_local_addr,
        grad_output_local_addr,
        input_local_addr,
        &input_shape,
        &aligned_stride,
        &aligned_stride,
        &aligned_stride,
        dtype);
    tpu_bdc_fp_scale(
        grad_output_local_addr,
        grad_output_local_addr,
        weight_local_addr,
        &input_shape,
        dtype);
    if(tpu_is_parallel_state())tpu_parallel_end();
    tpu_gdma_cpy_L2S(
        grad_input_global_addr,
        grad_output_local_addr,
        &input_shape,
        &global_stride,
        &aligned_stride,
        dtype);
    return;
}

void nodechip_bn_backward(
    global_addr_t grad_output_global_addr,
    global_addr_t input_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t saved_mean_global_addr,
    global_addr_t saved_invstd_global_addr,
    global_addr_t grad_input_global_addr,
    global_addr_t grad_weight_global_addr,
    global_addr_t grad_bias_global_addr,
    dim4 shape,
    data_type_t dtype,
    int grad_input_enable,
    int grad_weight_enable,
    int grad_biad_enable)
{
    TPUKERNEL_ASSERT(grad_input_enable);
    if(!is_local_mem_enough(shape.n, shape.c, shape.h*shape.w, 0, dtype))
    {
        if(is_local_mem_enough(shape.n, NPU_NUM, shape.h*shape.w, 0, dtype))
        {
            //cut c   
            int csecs = ALIGN(shape.c,NPU_NUM);
            while(!is_local_mem_enough(shape.n, csecs, shape.h*shape.w, 0, dtype))
            {
                csecs-=NPU_NUM;
            }
            int csecs_num = DIV_UP(shape.c, csecs);
            for(int cidx=0; cidx < csecs_num; cidx++)
            {
                bn_backward_split_c(
                    grad_output_global_addr + cidx * csecs * shape.h * shape.w * tpu_data_type_size(dtype),
                    input_global_addr + cidx * csecs * shape.h * shape.w * tpu_data_type_size(dtype),
                    weight_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    saved_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    saved_invstd_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    grad_input_global_addr + cidx * csecs * shape.h * shape.w * tpu_data_type_size(dtype),
                    grad_weight_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    grad_bias_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    shape,
                    cidx == csecs_num-1 ? shape.c - cidx * csecs : csecs,
                    dtype,
                    grad_weight_enable,
                    grad_biad_enable);
            }
        }
        else if(is_local_mem_enough(shape.n, shape.c, tpu_eu_num(dtype), 1, dtype))
        {   
            //cut hw
            int hwsecs = ALIGN(shape.h * shape.w, tpu_eu_num(dtype));
            while(!is_local_mem_enough(shape.n, shape.c, hwsecs, 1, dtype))
            {
                hwsecs-=tpu_eu_num(dtype);
            }
            int hwsecs_num = DIV_UP(shape.h * shape.w, hwsecs);
            for(int hwidx=0; hwidx < hwsecs_num; hwidx++)
            {
                accumulate_grad_weight_bias(
                    grad_output_global_addr + hwidx * hwsecs * tpu_data_type_size(dtype),
                    input_global_addr + hwidx * hwsecs * tpu_data_type_size(dtype),
                    saved_mean_global_addr,
                    saved_invstd_global_addr,
                    weight_global_addr,
                    shape,
                    hwidx == hwsecs_num-1 ? shape.h * shape.w - hwidx * hwsecs : hwsecs,
                    hwidx,
                    dtype);
            }
            hwsecs = ALIGN(shape.h * shape.w, tpu_eu_num(dtype));
            while(!is_local_mem_enough(shape.n, shape.c, hwsecs, 2, dtype))
            {
                hwsecs-=tpu_eu_num(dtype);
            }
            hwsecs_num = DIV_UP(shape.h * shape.w, hwsecs);
            for(int hwidx=0; hwidx < hwsecs_num; hwidx++)
            {
                bn_backward_split_hw(
                    grad_output_global_addr + hwidx * hwsecs * tpu_data_type_size(dtype),
                    input_global_addr + hwidx * hwsecs * tpu_data_type_size(dtype),
                    grad_input_global_addr + hwidx * hwsecs * tpu_data_type_size(dtype),
                    grad_weight_global_addr,
                    grad_bias_global_addr,
                    shape,
                    hwidx == hwsecs_num-1 ? shape.h * shape.w - hwidx * hwsecs : hwsecs,
                    hwidx,
                    dtype,
                    grad_weight_enable,
                    grad_biad_enable);
            }
        }
        else
        {   //TODO: cut chw/n
            TPUKERNEL_ASSERT(is_local_mem_enough(shape.n, NPU_NUM, tpu_eu_num(dtype), 0, dtype));         
            return;
        }
    }
    else
    {
        bn_backward_split_c(
            grad_output_global_addr,
            input_global_addr,
            weight_global_addr,
            saved_mean_global_addr,
            saved_invstd_global_addr,
            grad_input_global_addr,
            grad_weight_global_addr,
            grad_bias_global_addr,
            shape,
            shape.c,
            dtype,
            grad_weight_enable,
            grad_biad_enable);
    }
}

void tpu_kernel_api_batchnorm_backward(const void *args)
{
    sg_api_bn_backward_t *api = (sg_api_bn_backward_t *)args;
    dim4 shape = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};
    data_type_t dtype = DT_FP16;

    TPUKERNEL_ASSERT(is_local_mem_enough(shape.n, shape.c, 32, 1, dtype));
    TPUKERNEL_ASSERT(api->grad_input_enable);
    
    tpu_initialize();
    nodechip_bn_backward(
            api->grad_output_global_addr,
            api->input_global_addr,
            api->weight_global_addr,
            api->saved_mean_global_addr,
            api->saved_invstd_global_addr,
            api->grad_input_global_addr,
            api->grad_weight_global_addr,
            api->grad_bias_global_addr,
            shape,
            dtype,
            api->grad_input_enable,
            api->grad_weight_enable,
            api->grad_bias_enable);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_batchnorm_backward);