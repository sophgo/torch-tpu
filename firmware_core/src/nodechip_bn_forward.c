#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#include "tpu_defs.h"
/**
 * batchnorm_forward_training:
 * 1. output = ((input - batch_mean) / sqrt(batch_var + eps) * weight + bias
 * 2. running_stats = momentum * batch_stats + (1-momentum) * running_stats
 * 3. save batch stats for backward
 * 
 * NOTICE:
 * 1. compute forward/backward:  batch_var = Σ(input-mean)^2 /N,     (unbiased = false)
 * 2. update running_var:        batch_var = Σ(input-mean)^2 /(n-1), (unbiased = true)
 */

static inline bool is_local_mem_enough(
    int n,
    int c,
    int hw,
    bool split_c,
    data_type_t dtype)
{
    int c_param_size = DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype);
    int pooling_size = DIV_UP(c, NPU_NUM) * 64;
    int param_size = DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(n, hw, dtype);
    int total_size = split_c ? ALIGN(ALIGN(c_param_size * 6, 64) + pooling_size, BANK_SIZE)
                             + ALIGN(param_size, BANK_SIZE) * 2
                             : ALIGN(ALIGN(c_param_size * 4, 64) + pooling_size, BANK_SIZE)
                             + ALIGN(param_size, BANK_SIZE);
    return total_size < LOCAL_MEM_SIZE;
}

void batchnorm_forward_split_c(
    global_addr_t input_global_addr,
    global_addr_t running_mean_global_addr,
    global_addr_t running_var_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t bias_global_addr,
    global_addr_t updated_mean_global_addr,
    global_addr_t updated_var_global_addr,
    global_addr_t batch_mean_global_addr,
    global_addr_t batch_invstd_global_addr,
    global_addr_t output_global_addr,
    dim4          ori_shape,
    int           nhw,
    int           csecs,
    float         momentum,
    float         eps,
    data_type_t   dtype)
{
    dim4 shape = {1, csecs, 1, nhw};
    dim4 cshape = {1, csecs, 1, 1};
    dim4 aligned_stride;
    dim4 global_stride;
    dim4 compact_stride;
    tpu_aligned_stride(&aligned_stride, 0, &shape, dtype);
    tpu_continuous_stride(&global_stride, &ori_shape);
    tpu_compact_stride(&compact_stride, 0, &cshape);
    scalar_t scale = {.f32 = 1.f/nhw};
    scalar_t ep = {.f32 = eps};
    dim2 pooling_kernel = {1, nhw};
    dim2 pooling_stride = {1, 1};
    dim2 pooling_dilation = {1, 1};
    padding_t pooling_padding = {0, 0, 0, 0};

    int c_param_size = ALIGN(DIV_UP(csecs, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int pooling_size = DIV_UP(csecs, NPU_NUM) * 64;
    int param_size = DIV_UP(csecs,NPU_NUM) * tpu_aligned_feature_size(nhw, 1, DT_FP32);
    //param (1,c,1,1) compact
    local_addr_t batch_mean_local_addr = 0;
    local_addr_t batch_invstd_local_addr = batch_mean_local_addr + c_param_size;
    local_addr_t running_mean_local_addr = batch_invstd_local_addr + c_param_size;
    local_addr_t running_var_local_addr = running_mean_local_addr + c_param_size;
    local_addr_t weight_local_addr = running_var_local_addr + c_param_size;
    local_addr_t bias_local_addr = weight_local_addr + c_param_size;
    local_addr_t pooling_buffer = ALIGN(c_param_size * 6, 64);
    //param (n,c,h,w) aligned
    local_addr_t input_local_addr = ALIGN(pooling_buffer + pooling_size, BANK_SIZE);
    local_addr_t output_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);

    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &shape,
        &aligned_stride,
        &global_stride,
        dtype);
    tpu_gdma_compact_S2L(
        weight_local_addr,
        weight_global_addr,
        &cshape,
        dtype);
    tpu_gdma_compact_S2L(
        bias_local_addr,
        bias_global_addr,
        &cshape,
        dtype);
    tpu_gdma_compact_S2L(
        running_mean_local_addr,
        running_mean_global_addr,
        &cshape,
        dtype);
    tpu_gdma_compact_S2L(
        running_var_local_addr,
        running_var_global_addr,
        &cshape,
        dtype);
    tpu_bdc_fp_avg_pool2d(
        pooling_buffer,
        input_local_addr,
        &shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        dtype,
        scale);
    tpu_bdc_cpy(
        batch_mean_local_addr,
        pooling_buffer,
        &cshape,
        &compact_stride,
        NULL,
        dtype);
    tpu_gdma_compact_L2S(
        batch_mean_global_addr,
        batch_mean_local_addr,
        &cshape,
        dtype);
    tpu_bdc_fp_sub_bias_sqr(
        output_local_addr,
        input_local_addr,
        batch_mean_local_addr,
        &shape,
        dtype);
    scale.f32 = momentum/(nhw-1.f);
    tpu_bdc_fp_avg_pool2d(
        pooling_buffer,
        output_local_addr,
        &shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        dtype,
        scale);
    // update running stats
    scale.f32 = 1-momentum;
    tpu_bdc_fp_mul_C(
        running_mean_local_addr,
        running_mean_local_addr,
        scale,
        &cshape,
        &compact_stride,
        &compact_stride,
        dtype);
    tpu_bdc_fp_mul_C(
        running_var_local_addr,
        running_var_local_addr,
        scale,
        &cshape,
        &compact_stride,
        &compact_stride,
        dtype);
    tpu_bdc_fp32_mac_C(
        running_mean_local_addr,
        batch_mean_local_addr,
        momentum,
        &cshape,
        &compact_stride,
        &compact_stride);
    tpu_bdc_fp_add(
        running_var_local_addr,
        running_var_local_addr,
        pooling_buffer,
        &cshape,
        &compact_stride,
        &compact_stride,
        NULL,
        dtype);
    tpu_gdma_compact_L2S(
        updated_mean_global_addr,
        running_mean_local_addr,
        &cshape,
        dtype);
    tpu_gdma_compact_L2S(
        updated_var_global_addr,
        running_var_local_addr,
        &cshape,
        dtype);
    // save batch stats
    scale.f32 = (nhw-1)/(nhw*momentum);
    tpu_bdc_fp_scale_bias_C(
        pooling_buffer,
        pooling_buffer,
        scale,
        ep,
        &cshape,
        dtype);
    tpu_bdc_fp32_rsqrt(
        pooling_buffer,
        pooling_buffer,
        &cshape);
    tpu_bdc_cpy(
        batch_invstd_local_addr,
        pooling_buffer,
        &cshape,
        &compact_stride,
        NULL,
        dtype);
    tpu_gdma_compact_L2S(
        batch_invstd_global_addr,
        batch_invstd_local_addr,
        &cshape,
        dtype);
    tpu_bdc_neg(
        batch_mean_local_addr,
        batch_mean_local_addr,
        &cshape,
        &compact_stride,
        &compact_stride,
        dtype);
    tpu_bdc_fp_bias(
        input_local_addr,
        input_local_addr,
        batch_mean_local_addr,
        &shape,
        dtype);
    tpu_bdc_fp_scale(
        input_local_addr,
        input_local_addr,
        batch_invstd_local_addr,
        &shape,
        dtype);
    tpu_bdc_fp_scale_bias(
        input_local_addr,
        input_local_addr,
        weight_local_addr,
        bias_local_addr,
        &shape,
        dtype);
    tpu_gdma_cpy_L2S(
        output_global_addr,
        input_local_addr,
        &shape,
        &global_stride,
        &aligned_stride,
        dtype);
}

void batchnorm_forward_split_hw(
    global_addr_t input_global_addr,
    global_addr_t running_mean_global_addr,
    global_addr_t running_var_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t bias_global_addr,
    global_addr_t updated_mean_global_addr,
    global_addr_t updated_var_global_addr,
    global_addr_t batch_mean_global_addr,
    global_addr_t batch_invstd_global_addr,
    global_addr_t output_global_addr,
    dim4          ori_shape,
    float         momentum,
    float         eps,
    data_type_t   dtype)
{
    int c = ori_shape.c;
    int nhw = ori_shape.w;
    int c_param_size = ALIGN(DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int pooling_size = DIV_UP(c, NPU_NUM) * 64;

    local_addr_t batch_mean_local_addr = 0;
    local_addr_t batch_var_local_addr = batch_mean_local_addr + c_param_size;
    local_addr_t running_mean_local_addr = batch_var_local_addr + c_param_size;
    local_addr_t running_var_local_addr = running_mean_local_addr + c_param_size;
    local_addr_t bias_local_addr = running_mean_local_addr;
    local_addr_t weight_local_addr = running_var_local_addr;
    local_addr_t pooling_buffer = ALIGN(c_param_size * 4, 64);
    local_addr_t input_local_addr = ALIGN(pooling_buffer + pooling_size, BANK_SIZE);

    int nhw_secs = ALIGN(nhw, tpu_eu_num(dtype))-tpu_eu_num(dtype);
    while(!is_local_mem_enough(1, c, nhw_secs, 0, dtype))
    {
        nhw_secs-=tpu_eu_num(dtype);
    }
    int nhwslice = DIV_UP(nhw, nhw_secs);
    for(int nhwidx=0; nhwidx < 3 * nhwslice; nhwidx++)
    {
        bool if_last = (nhwidx == nhwslice-1)||(nhwidx == 2*nhwslice-1)||(nhwidx == 3*nhwslice-1);
        int nhwsecs = if_last ? nhw - (nhwslice-1) * nhw_secs : nhw_secs;
        dim4 shape = {1, c, 1, nhwsecs};
        dim4 cshape = {1, c, 1, 1};
        dim4 aligned_stride;
        dim4 global_stride;
        dim4 compact_stride;
        tpu_aligned_stride(&aligned_stride, 0, &shape, dtype);
        tpu_continuous_stride(&global_stride, &ori_shape);
        tpu_compact_stride(&compact_stride, 0, &cshape);
        scalar_t scale = {.f32 = 0};
        dim2 pooling_kernel = {1, nhwsecs};
        dim2 pooling_stride = {1, 1};
        dim2 pooling_dilation = {1, 1};
        padding_t pooling_padding = {0, 0, 0, 0};

        if(!nhwidx)
        {
            tpu_bdc_set_C(
                batch_mean_local_addr,
                scale,
                &cshape,
                &compact_stride,
                dtype);
            tpu_bdc_set_C(
                batch_var_local_addr,
                scale,
                &cshape,
                &compact_stride,
                dtype);    
            tpu_gdma_compact_S2L(
                running_mean_local_addr,
                running_mean_global_addr,
                &cshape,
                dtype);
            tpu_gdma_compact_S2L(
                running_var_local_addr,
                running_var_global_addr,
                &cshape,
                dtype);
        }
        if(nhwidx < nhwslice)
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + nhwidx * nhw_secs * tpu_data_type_size(dtype),
                &shape,
                &aligned_stride,
                &global_stride,
                dtype);
            scale.f32 = 1.f/nhw;
            tpu_bdc_fp_avg_pool2d(
                pooling_buffer,
                input_local_addr,
                &shape,
                &pooling_kernel,
                &pooling_padding,
                &pooling_stride,
                &pooling_dilation,
                dtype,
                scale);
            tpu_bdc_fp_add(
                batch_mean_local_addr,
                batch_mean_local_addr,
                pooling_buffer,
                &cshape,
                &compact_stride,
                &compact_stride,
                NULL,
                dtype);
            if(nhwidx==nhwslice-1)
            {
                tpu_gdma_compact_L2S(
                    batch_mean_global_addr,
                    batch_mean_local_addr,
                    &cshape,
                    dtype);
            }
        }
        else if(nhwidx < 2 * nhwslice)
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + (nhwidx - nhwslice) * nhw_secs * tpu_data_type_size(dtype),
                &shape,
                &aligned_stride,
                &global_stride,
                dtype);
            tpu_bdc_fp_sub_bias_sqr(
                input_local_addr,
                input_local_addr,
                batch_mean_local_addr,
                &shape,
                dtype);
            scale.f32 = momentum/(nhw-1);
            tpu_bdc_fp_avg_pool2d(
                pooling_buffer,
                input_local_addr,
                &shape,
                &pooling_kernel,
                &pooling_padding,
                &pooling_stride,
                &pooling_dilation,
                dtype,
                scale);
            tpu_bdc_fp_add(
                batch_var_local_addr,
                batch_var_local_addr,
                pooling_buffer,
                &cshape,
                &compact_stride,
                &compact_stride,
                NULL,
                dtype);
            if(nhwidx == nhwslice * 2 - 1)
            {
                scale.f32 = 1-momentum;
                tpu_bdc_fp_mul_C(
                    running_mean_local_addr,
                    running_mean_local_addr,
                    scale,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    dtype);
                tpu_bdc_fp_mul_C(
                    running_var_local_addr,
                    running_var_local_addr,
                    scale,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    dtype);
                tpu_bdc_fp32_mac_C(
                    running_mean_local_addr,
                    batch_mean_local_addr,
                    momentum,
                    &cshape,
                    &compact_stride,
                    &compact_stride);
                tpu_bdc_fp_add(
                    running_var_local_addr,
                    running_var_local_addr,
                    batch_var_local_addr,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    &compact_stride,
                    dtype);
                tpu_gdma_compact_L2S(
                    updated_mean_global_addr,
                    running_mean_local_addr,
                    &cshape,
                    dtype);
                tpu_gdma_compact_L2S(
                    updated_var_global_addr,
                    running_var_local_addr,
                    &cshape,
                    dtype);
                scale.f32 = (nhw-1)/(nhw*momentum);
                tpu_bdc_fp_mul_C(
                    batch_var_local_addr,
                    batch_var_local_addr,
                    scale,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    dtype);
                scale.f32 = eps;
                tpu_bdc_fp_add_C(
                    pooling_buffer,
                    batch_var_local_addr,
                    scale,
                    &cshape,
                    NULL,
                    &compact_stride,
                    dtype);
                tpu_bdc_fp32_rsqrt(
                    pooling_buffer,
                    pooling_buffer,
                    &cshape);
                tpu_gdma_cpy_L2S(
                    batch_invstd_global_addr,
                    pooling_buffer,
                    &cshape,
                    NULL,
                    NULL,
                    dtype);
                tpu_bdc_cpy(
                    batch_var_local_addr,
                    pooling_buffer,
                    &cshape,
                    &compact_stride,
                    NULL,
                    dtype);
            }
        }
        else
        {
            if(nhwidx==2*nhwslice)
            {
                tpu_gdma_compact_S2L(
                    weight_local_addr,
                    weight_global_addr,
                    &cshape,
                    dtype);
                tpu_gdma_compact_S2L(
                    bias_local_addr,
                    bias_global_addr,
                    &cshape,
                    dtype);
                tpu_bdc_neg(
                    batch_mean_local_addr,
                    batch_mean_local_addr,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    dtype);
            }
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + (nhwidx - 2 * nhwslice) * nhw_secs * tpu_data_type_size(dtype),
                &shape,
                &aligned_stride,
                &global_stride,
                dtype);
            tpu_bdc_fp_bias(
                input_local_addr,
                input_local_addr,
                batch_mean_local_addr,
                &shape,
                dtype);
            tpu_bdc_fp_scale(
                input_local_addr,
                input_local_addr,
                batch_var_local_addr,
                &shape,
                dtype);
            tpu_bdc_fp_scale_bias(
                input_local_addr,
                input_local_addr,
                weight_local_addr,
                bias_local_addr,
                &shape,
                dtype);
            tpu_gdma_cpy_L2S(
                output_global_addr + (nhwidx - 2 * nhwslice) * nhw_secs * tpu_data_type_size(dtype),
                input_local_addr,
                &shape,
                &global_stride,
                &aligned_stride,
                dtype);
        }
    }
}

void nodechip_batchnorm_forward_training(
    global_addr_t input_global_addr,
    global_addr_t running_mean_global_addr,
    global_addr_t running_var_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t bias_global_addr,
    global_addr_t updated_mean_global_addr,
    global_addr_t updated_var_global_addr,
    global_addr_t batch_mean_global_addr,
    global_addr_t batch_invstd_global_addr,
    global_addr_t output_global_addr,
    dim4          shape,
    float         momentum,
    float         eps,
    data_type_t   dtype)
{
    dim4 trans_shape = {shape.c, shape.n, shape.h, shape.w};
    dim4 new_shape = {1, shape.c, 1, shape.n*shape.h*shape.w};
    tpu_gdma_cpy_nc_trans_S2S(
        output_global_addr,
        input_global_addr,
        &trans_shape,
        NULL,
        NULL,
        dtype);
    if(!is_local_mem_enough(shape.n, shape.c, shape.h*shape.w, 1, dtype))
    {
        if(is_local_mem_enough(shape.n, NPU_NUM, shape.h*shape.w, 1, dtype))
        {
            int csecs = ALIGN(shape.c,NPU_NUM)-NPU_NUM;
            while(!is_local_mem_enough(shape.n, csecs, shape.h*shape.w, 1, dtype))
            {
                csecs-=NPU_NUM;
            }
            int cslice = DIV_UP(shape.c, csecs);
            for(int cidx=0; cidx < cslice; cidx++)
            {
                batchnorm_forward_split_c(
                    output_global_addr + cidx * csecs * shape.n * shape.h * shape.w * tpu_data_type_size(dtype),
                    running_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    running_var_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    weight_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    bias_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    updated_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    updated_var_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    batch_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    batch_invstd_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    input_global_addr + cidx * csecs * shape.n * shape.h * shape.w * tpu_data_type_size(dtype),
                    new_shape,
                    shape.n*shape.h*shape.w,
                    cidx == cslice-1 ? shape.c - cidx * csecs : csecs,
                    momentum,
                    eps,
                    dtype);
            }
        }
        else
        {   
            TPUKERNEL_ASSERT(is_local_mem_enough(shape.n, shape.c, tpu_eu_num(dtype), 0, dtype));
            batchnorm_forward_split_hw(
                output_global_addr,
                running_mean_global_addr,
                running_var_global_addr,
                weight_global_addr,
                bias_global_addr,
                updated_mean_global_addr,
                updated_var_global_addr,
                batch_mean_global_addr,
                batch_invstd_global_addr,
                input_global_addr,
                new_shape,
                momentum,
                eps,
                dtype);
        }
    }
    else
    {
        batchnorm_forward_split_c(
            output_global_addr,
            running_mean_global_addr,
            running_var_global_addr,
            weight_global_addr,
            bias_global_addr,
            updated_mean_global_addr,
            updated_var_global_addr,
            batch_mean_global_addr,
            batch_invstd_global_addr,
            input_global_addr,
            new_shape,
            shape.n*shape.h*shape.w,
            shape.c,
            momentum,
            eps,
            dtype);
    }
    tpu_gdma_cpy_nc_trans_S2S(
        output_global_addr,
        input_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
}

void tpu_kernel_api_batchnorm_forward(const void* args)
{
    sg_api_batchnorm_forward_t *api = (sg_api_batchnorm_forward_t *)args;
    dim4 shape = {api->shape[0],api->shape[1],api->shape[2],api->shape[3]};
    
    TPUKERNEL_ASSERT(api->dtype==SG_DTYPE_FP32);
    tpu_initialize();
    nodechip_batchnorm_forward_training(
        api->input_global_addr,
        api->running_mean_global_addr,
        api->running_var_global_addr,
        api->weight_global_addr,
        api->bias_global_addr,
        api->updated_mean_global_addr,
        api->updated_var_global_addr,
        api->batch_mean_global_addr,
        api->batch_invstd_global_addr,
        api->output_global_addr,
        shape,
        api->momentum,
        api->eps,
        DT_FP32);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_batchnorm_forward);