#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

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
    bool split_n,
    data_type_t dtype)
{
    int c_param_size = DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype);
    int param_size = n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(1, hw, dtype);
    int c_pooling_size = DIV_UP(c, NPU_NUM) * 64;
    int pooling_size = n > 1 ? DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(1, n, dtype) : 0;
    int total_size = split_c ? ALIGN(c_param_size * 7, BANK_SIZE)
                             + ALIGN(c_pooling_size + pooling_size, BANK_SIZE)
                             + ALIGN(param_size, BANK_SIZE) * 2:
                     split_n ? ALIGN(c_param_size * 4, BANK_SIZE)
                             + ALIGN(c_pooling_size + pooling_size, BANK_SIZE)
                             + ALIGN(param_size, BANK_SIZE)
                             : ALIGN(c_param_size * 4, BANK_SIZE)
                             + ALIGN(c_pooling_size + pooling_size, BANK_SIZE)
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
    int           csecs,
    float         momentum,
    float         eps,
    data_type_t   dtype)
{
    int n = ori_shape.n;
    int h = ori_shape.h;
    int w = ori_shape.w;
    dim4 input_shape = {n, csecs, h, w};
    dim4 pooling_shape = {1, csecs, n, ALIGN(h * w, tpu_eu_num(dtype))};
    dim4 cshape = {1, csecs, 1, 1};
    dim4 compact_stride;
    dim4 aligned_stride;
    dim4 global_stride;
    dim4 zero_stride = {0, 1, 0, 0};
    tpu_compact_stride(&compact_stride, 0, &cshape);
    tpu_aligned_stride(&aligned_stride, 0, &input_shape, dtype);
    tpu_continuous_stride(&global_stride ,&ori_shape);
    dim4 local_stride = aligned_stride;
    if(csecs > NPU_NUM)
    {
        local_stride.n = local_stride.c;
        local_stride.c *= n;
    }
    scalar_t scale = {.f32 = 0};
    scalar_t ep = {.f32 = eps};
    dim2 pooling_kernel = {1, pooling_shape.w};
    dim2 pooling_stride = {1, 1};
    dim2 pooling_dilation = {1, 1};
    padding_t pooling_padding = {0, 0, 0, 0};
    int c_param_size = ALIGN(DIV_UP(csecs, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int param_size = n * DIV_UP(csecs, NPU_NUM) * tpu_aligned_feature_size(h, w, dtype);
    int c_pooling_size = DIV_UP(csecs, NPU_NUM) * 64;
    int pooling_size = n > 1 ? DIV_UP(csecs, NPU_NUM) * tpu_aligned_feature_size(1, n, dtype) : 0;
    
    //param (1,c,1,1) compact
    local_addr_t batch_mean_local_addr = 0;
    local_addr_t batch_var_local_addr = batch_mean_local_addr + c_param_size;
    local_addr_t batch_invstd_local_addr = batch_var_local_addr + c_param_size;
    local_addr_t running_mean_local_addr = batch_invstd_local_addr + c_param_size;
    local_addr_t running_var_local_addr = running_mean_local_addr + c_param_size;
    local_addr_t weight_local_addr = running_var_local_addr + c_param_size;
    local_addr_t bias_local_addr = weight_local_addr + c_param_size;
    //pooling buffer aligned
    local_addr_t c_pooling_buffer = ALIGN(c_param_size * 7, BANK_SIZE);
    local_addr_t pooling_buffer = n > 1 ? c_pooling_buffer + c_pooling_size : c_pooling_buffer;
    //param (n,c,h,w) aligned
    local_addr_t input_local_addr = c_pooling_buffer + ALIGN(c_pooling_size + pooling_size, BANK_SIZE);
    local_addr_t output_local_addr = input_local_addr + ALIGN(param_size, BANK_SIZE);
    if((h * w) % (tpu_eu_num(DT_FP32)))
    {
        tpu_parallel_start();
        tpu_bdc_set_C(
            input_local_addr,
            scale,
            &pooling_shape,
            NULL,
            dtype);
        tpu_bdc_set_C(
            output_local_addr,
            scale,
            &pooling_shape,
            NULL,
            dtype);
    }
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
    if(tpu_is_parallel_state()){tpu_parallel_end();}
    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        &global_stride,
        dtype);
    scale.f32 = 1.f/(h*w);
    tpu_bdc_fp_avg_pool2d(
        pooling_buffer,
        input_local_addr,
        &pooling_shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        dtype,
        scale);
    if(n>1)
    {
        pooling_shape.w = pooling_shape.h;
        pooling_shape.h = 1;
        pooling_kernel.w = pooling_shape.w;
        scale.f32 = 1.f/n;
        tpu_bdc_fp_avg_pool2d(
            c_pooling_buffer,
            pooling_buffer,
            &pooling_shape,
            &pooling_kernel,
            &pooling_padding,
            &pooling_stride,
            &pooling_dilation,
            dtype,
            scale);
        pooling_shape.h = n;
        pooling_shape.w = ALIGN(h * w, tpu_eu_num(dtype));
        pooling_kernel.w = pooling_shape.w;
    }
    tpu_bdc_cpy(
        batch_mean_local_addr,
        c_pooling_buffer,
        &cshape,
        &compact_stride,
        NULL,
        dtype);
    tpu_bdc_fp_sub(
        output_local_addr,
        input_local_addr,
        batch_mean_local_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        &zero_stride,
        dtype);
    tpu_bdc_fp_mul(
        output_local_addr,
        output_local_addr,
        output_local_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        dtype);
    scale.f32 = 1.f;
    tpu_bdc_fp_avg_pool2d(
        pooling_buffer,
        output_local_addr,
        &pooling_shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        dtype,
        scale);
    if(n>1)
    {
        pooling_shape.w = pooling_shape.h;
        pooling_shape.h = 1;
        pooling_kernel.w = pooling_shape.w;
        tpu_bdc_fp_avg_pool2d(
            c_pooling_buffer,
            pooling_buffer,
            &pooling_shape,
            &pooling_kernel,
            &pooling_padding,
            &pooling_stride,
            &pooling_dilation,
            dtype,
            scale);
    }
    tpu_bdc_cpy(
        batch_var_local_addr,
        c_pooling_buffer,
        &cshape,
        &compact_stride,
        NULL,
        dtype);
    // update running stats
    scale.f32 = 1.f-momentum;
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
    tpu_bdc_fp32_mac_C(
        running_var_local_addr,
        batch_var_local_addr,
        momentum/(n*h*w-1.f),
        &cshape,
        &compact_stride,
        &compact_stride);
    tpu_parallel_start();
    tpu_gdma_compact_L2S(
        batch_mean_global_addr,
        batch_mean_local_addr,
        &cshape,
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
    scale.f32 = 1.f/(n*h*w);
    tpu_bdc_fp_scale_bias_C(
        c_pooling_buffer,
        c_pooling_buffer,
        scale,
        ep,
        &cshape,
        dtype);
    tpu_bdc_fp32_rsqrt(
        c_pooling_buffer,
        c_pooling_buffer,
        &cshape);
    tpu_bdc_cpy(
        batch_invstd_local_addr,
        c_pooling_buffer,
        &cshape,
        &compact_stride,
        NULL,
        dtype);
    tpu_bdc_fp_sub(
        input_local_addr,
        input_local_addr,
        batch_mean_local_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        &zero_stride,
        dtype);
    tpu_bdc_fp_mul(
        input_local_addr,
        input_local_addr,
        batch_invstd_local_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        &zero_stride,
        dtype);
    tpu_parallel_end();
    tpu_parallel_start();
    tpu_gdma_compact_L2S(
        batch_invstd_global_addr,
        batch_invstd_local_addr,
        &cshape,
        dtype);
    tpu_bdc_fp_mul(
        input_local_addr,
        input_local_addr,
        weight_local_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        &zero_stride,
        dtype);
    tpu_bdc_fp_add(
        input_local_addr,
        input_local_addr,
        bias_local_addr,
        &input_shape,
        csecs > 64 ? &local_stride : NULL,
        csecs > 64 ? &local_stride : NULL,
        &zero_stride,
        dtype);
    tpu_parallel_end();
    tpu_gdma_cpy_L2S(
        output_global_addr,
        input_local_addr,
        &input_shape,
        &global_stride,
        csecs > 64 ? &local_stride : NULL,
        dtype);
}

void batchnorm_forward_split_n(
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
    int n = ori_shape.n;
    int c = ori_shape.c;
    int hw = ori_shape.h * ori_shape.w;
    int n_secs = n-1;
    int c_param_size = ALIGN(DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int c_pooling_size = DIV_UP(c, NPU_NUM) * 64;
    TPUKERNEL_ASSERT(!(hw%tpu_eu_num(dtype)));
    local_addr_t batch_mean_local_addr = 0;
    local_addr_t batch_var_local_addr = batch_mean_local_addr + c_param_size;
    local_addr_t running_mean_local_addr = batch_var_local_addr + c_param_size;
    local_addr_t running_var_local_addr = running_mean_local_addr + c_param_size;
    local_addr_t bias_local_addr = running_mean_local_addr;
    local_addr_t weight_local_addr = running_var_local_addr;
    local_addr_t c_pooling_buffer = ALIGN(c_param_size * 4, BANK_SIZE);
    while(!is_local_mem_enough(n_secs, c, hw, 0, 1, dtype))
    {
        n_secs-=1;
    }
    int nslice = DIV_UP(ori_shape.n, n_secs);    
    
    // loop 1&2: accumulate batch_stats
    // loop  3 : compute output
    for(int nidx=0; nidx < 3 * nslice; nidx++)
    {
        bool if_last = (nidx == nslice-1)||(nidx == 2 * nslice-1)||(nidx == 3 * nslice-1);
        int nsecs = if_last ? n - (nslice-1) * n_secs:n_secs; 
        int pooling_size = nsecs > 1 ? DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(1, nsecs, dtype) : 0;
        local_addr_t pooling_buffer = nsecs > 1 ? c_pooling_buffer + c_pooling_size : c_pooling_buffer;
        local_addr_t input_local_addr = c_pooling_buffer + ALIGN(c_pooling_size + pooling_size, BANK_SIZE);
        dim4 input_shape = {nsecs, c, 1, hw};
        dim4 pooling_shape = {1, c, nsecs, hw};
        dim4 cshape = {1, c, 1, 1};
        dim4 local_stride;
        dim4 global_stride;
        dim4 compact_stride;
        dim4 zero_stride = {0, 1, 0, 0};
        tpu_aligned_stride(&local_stride, 0, &input_shape, dtype);
        tpu_continuous_stride(&global_stride, &ori_shape);
        tpu_compact_stride(&compact_stride, 0, &cshape);
        if(c > NPU_NUM)
        {
            local_stride.n = local_stride.c;
            local_stride.c *= nsecs;
        }
        scalar_t scale = {.f32 = 0.f};
        scalar_t ep = {.f32 = eps};
        dim2 pooling_kernel = {1, pooling_shape.w};
        dim2 pooling_stride = {1, 1};
        dim2 pooling_dilation = {1, 1};
        padding_t pooling_padding = {0, 0, 0, 0};
        if(!nidx)
        {
            tpu_parallel_start();
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
        if(nidx < nslice)
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + nidx * n_secs * c * hw * tpu_data_type_size(dtype),
                &input_shape,
                c > 64 ? &local_stride : NULL,
                NULL,//&global_stride,
                dtype);
            if(tpu_is_parallel_state()){tpu_parallel_end();}
            scale.f32 = 1.f;
            tpu_bdc_fp_avg_pool2d(
                pooling_buffer,
                input_local_addr,
                &pooling_shape,
                &pooling_kernel,
                &pooling_padding,
                &pooling_stride,
                &pooling_dilation,
                dtype,
                scale);
            if(nsecs>1)
            {
                pooling_shape.w = pooling_shape.h;
                pooling_shape.h = 1;
                pooling_kernel.w = pooling_shape.w;
                tpu_bdc_fp_avg_pool2d(
                    c_pooling_buffer,
                    pooling_buffer,
                    &pooling_shape,
                    &pooling_kernel,
                    &pooling_padding,
                    &pooling_stride,
                    &pooling_dilation,
                    dtype,
                    scale);
            }
            tpu_parallel_start();
            tpu_bdc_fp_add(
                batch_mean_local_addr,
                batch_mean_local_addr,
                c_pooling_buffer,
                &cshape,
                &compact_stride,
                &compact_stride,
                NULL,
                dtype);
            if(nidx==nslice-1)
            {
                scale.f32 = 1.f/(n*hw);
                tpu_bdc_fp_mul_C(
                    batch_mean_local_addr,
                    batch_mean_local_addr,
                    scale,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    dtype);
                if(tpu_is_parallel_state()){tpu_parallel_end();}
                tpu_gdma_compact_L2S(
                    batch_mean_global_addr,
                    batch_mean_local_addr,
                    &cshape,
                    dtype);
            }
        }
        else if(nidx < 2 * nslice)
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + (nidx - nslice) * n_secs * c * hw * tpu_data_type_size(dtype),
                &input_shape,
                c > 64 ? &local_stride : NULL,
                NULL,//&global_stride,
                dtype);
            if(tpu_is_parallel_state()){tpu_parallel_end();}
            tpu_bdc_fp_sub(
                input_local_addr,
                input_local_addr,
                batch_mean_local_addr,
                &input_shape,
                c > 64 ? &local_stride : NULL,
                c > 64 ? &local_stride : NULL,
                &zero_stride,
                dtype);
            tpu_bdc_fp_mul(
                input_local_addr,
                input_local_addr,
                input_local_addr,
                &input_shape,
                c > 64 ? &local_stride : NULL,
                c > 64 ? &local_stride : NULL,
                c > 64 ? &local_stride : NULL,
                dtype);
            scale.f32 = 1.f;
            tpu_bdc_fp_avg_pool2d(
                pooling_buffer,
                input_local_addr,
                &pooling_shape,
                &pooling_kernel,
                &pooling_padding,
                &pooling_stride,
                &pooling_dilation,
                dtype,
                scale);
            if(nsecs>1)
            {
                pooling_shape.w = pooling_shape.h;
                pooling_shape.h = 1;
                pooling_kernel.w = pooling_shape.w;
                tpu_bdc_fp_avg_pool2d(
                    c_pooling_buffer,
                    pooling_buffer,
                    &pooling_shape,
                    &pooling_kernel,
                    &pooling_padding,
                    &pooling_stride,
                    &pooling_dilation,
                    dtype,
                    scale);
            }
            tpu_parallel_start();
            tpu_bdc_fp_add(
                batch_var_local_addr,
                batch_var_local_addr,
                c_pooling_buffer,
                &cshape,
                &compact_stride,
                &compact_stride,
                NULL,
                dtype);
            if(nidx == 2 * nslice - 1)
            {
                if(tpu_is_parallel_state()){tpu_parallel_end();}
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
                tpu_bdc_fp32_mac_C(
                    running_var_local_addr,
                    batch_var_local_addr,
                    momentum/(n*hw-1.f),
                    &cshape,
                    &compact_stride,
                    &compact_stride);
                tpu_parallel_start();
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
                tpu_bdc_cpy(
                    c_pooling_buffer,
                    batch_var_local_addr,
                    &cshape,
                    NULL,
                    &compact_stride,
                    dtype);
                scale.f32 = 1.f/(n*hw);
                tpu_bdc_fp_scale_bias_C(
                    c_pooling_buffer,
                    c_pooling_buffer,
                    scale,
                    ep,
                    &cshape,
                    dtype);
                tpu_bdc_fp32_rsqrt(
                    c_pooling_buffer,
                    c_pooling_buffer,
                    &cshape);
                tpu_bdc_cpy(
                    batch_var_local_addr,
                    c_pooling_buffer,
                    &cshape,
                    &compact_stride,
                    NULL,
                    dtype);
                if(tpu_is_parallel_state()){tpu_parallel_end();}
                tpu_gdma_compact_L2S(
                    batch_invstd_global_addr,
                    batch_var_local_addr,
                    &cshape,
                    dtype);
            }
        }
        else
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + (nidx - 2 * nslice) * n_secs * c * hw * tpu_data_type_size(dtype),
                &input_shape,
                NULL,
                NULL,//&global_stride,
                dtype);
            tpu_bdc_fp_bias(
                input_local_addr,
                input_local_addr,
                batch_mean_local_addr,
                &input_shape,
                dtype);
            tpu_bdc_fp_scale(
                input_local_addr,
                input_local_addr,
                batch_var_local_addr,
                &input_shape,
                dtype);
            tpu_bdc_fp_scale_bias(
                input_local_addr,
                input_local_addr,
                weight_local_addr,
                bias_local_addr,
                &input_shape,
                dtype);
            tpu_gdma_cpy_L2S(
                output_global_addr + (nidx - 2 * nslice) * n_secs * c * hw * tpu_data_type_size(dtype),
                input_local_addr,
                &input_shape,
                NULL,//&global_stride,
                NULL,
                dtype);
        }
    }
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
    int n = ori_shape.n;
    int c = ori_shape.c;
    int hw = ori_shape.h * ori_shape.w;
    int c_param_size = ALIGN(DIV_UP(c, NPU_NUM) * tpu_data_type_size(dtype), 4);
    int c_pooling_size = DIV_UP(c, NPU_NUM) * 64;
    int pooling_size = n > 1 ? DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(1, n,dtype) : 0;

    local_addr_t batch_mean_local_addr = 0;
    local_addr_t batch_var_local_addr = batch_mean_local_addr + c_param_size;
    local_addr_t running_mean_local_addr = batch_var_local_addr + c_param_size;
    local_addr_t running_var_local_addr = running_mean_local_addr + c_param_size;
    local_addr_t bias_local_addr = running_mean_local_addr;
    local_addr_t weight_local_addr = running_var_local_addr;
    local_addr_t c_pooling_buffer = ALIGN(c_param_size * 4, BANK_SIZE);
    local_addr_t pooling_buffer = n > 1 ? c_pooling_buffer + c_pooling_size : c_pooling_buffer;
    local_addr_t input_local_addr = c_pooling_buffer + ALIGN(c_pooling_size + pooling_size, BANK_SIZE);
    
    // loop 1&2: accumulate batch_stats
    // loop  3 : compute output
    int hw_secs = ALIGN(hw, tpu_eu_num(dtype))-tpu_eu_num(dtype);
    while(!is_local_mem_enough(n, c, hw_secs, 0, 0, dtype))
    {
        hw_secs-=tpu_eu_num(dtype);
    }
    int hwslice = DIV_UP(hw, hw_secs);
    for(int hwidx=0; hwidx < 3 * hwslice; hwidx++)
    {
        bool if_last = (hwidx == hwslice-1)||(hwidx == 2 * hwslice-1)||(hwidx == 3 * hwslice-1);
        int hwsecs = if_last ? hw - (hwslice-1) * hw_secs : hw_secs;
        TPUKERNEL_ASSERT(!(hwsecs%tpu_eu_num(dtype)));
        dim4 input_shape = {n, c, 1, hwsecs};
        dim4 pooling_shape = {1, c, n, hwsecs};
        dim4 cshape = {1, c, 1, 1};
        dim4 local_stride;
        dim4 global_stride;
        dim4 compact_stride;
        dim4 zero_stride = {0, 1, 0, 0};
        tpu_aligned_stride(&local_stride, 0, &input_shape, dtype);
        tpu_continuous_stride(&global_stride, &ori_shape);
        tpu_compact_stride(&compact_stride, 0, &cshape);
        if(c > NPU_NUM)
        {
            local_stride.n = local_stride.c;
            local_stride.c *= n;
        }
        scalar_t scale = {.f32 = 0.f};
        scalar_t ep = {.f32 = eps};
        dim2 pooling_kernel = {1, pooling_shape.w};
        dim2 pooling_stride = {1, 1};
        dim2 pooling_dilation = {1, 1};
        padding_t pooling_padding = {0, 0, 0, 0};
        if(!hwidx)
        {
            tpu_parallel_start();
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
        if(hwidx < hwslice)
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + hwidx * hw_secs * tpu_data_type_size(dtype),
                &input_shape,
                c > 64 ? &local_stride : NULL,
                &global_stride,
                dtype);
            if(tpu_is_parallel_state()){tpu_parallel_end();}
            scale.f32 = 1.f;
            tpu_bdc_fp_avg_pool2d(
                pooling_buffer,
                input_local_addr,
                &pooling_shape,
                &pooling_kernel,
                &pooling_padding,
                &pooling_stride,
                &pooling_dilation,
                dtype,
                scale);
            if(n>1)
            {
                pooling_shape.w = pooling_shape.h;
                pooling_shape.h = 1;
                pooling_kernel.w = pooling_shape.w;
                tpu_bdc_fp_avg_pool2d(
                    c_pooling_buffer,
                    pooling_buffer,
                    &pooling_shape,
                    &pooling_kernel,
                    &pooling_padding,
                    &pooling_stride,
                    &pooling_dilation,
                    dtype,
                    scale);
            }
            tpu_parallel_start();
            tpu_bdc_fp_add(
                batch_mean_local_addr,
                batch_mean_local_addr,
                c_pooling_buffer,
                &cshape,
                &compact_stride,
                &compact_stride,
                NULL,
                dtype);
            if(hwidx==hwslice-1)
            {
                scale.f32 = 1.f/(n*hw);
                tpu_bdc_fp_mul_C(
                    batch_mean_local_addr,
                    batch_mean_local_addr,
                    scale,
                    &cshape,
                    &compact_stride,
                    &compact_stride,
                    dtype);
                if(tpu_is_parallel_state()){tpu_parallel_end();}
                tpu_gdma_compact_L2S(
                    batch_mean_global_addr,
                    batch_mean_local_addr,
                    &cshape,
                    dtype);
            }
        }
        else if(hwidx < 2 * hwslice)
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + (hwidx - hwslice) * hw_secs * tpu_data_type_size(dtype),
                &input_shape,
                c > 64 ? &local_stride : NULL,
                &global_stride,
                dtype);
            if(tpu_is_parallel_state()){tpu_parallel_end();}
            tpu_bdc_fp_sub(
                input_local_addr,
                input_local_addr,
                batch_mean_local_addr,
                &input_shape,
                c > 64 ? &local_stride : NULL,
                c > 64 ? &local_stride : NULL,
                &zero_stride,
                dtype);
            tpu_bdc_fp_mul(
                input_local_addr,
                input_local_addr,
                input_local_addr,
                &input_shape,
                c > 64 ? &local_stride : NULL,
                c > 64 ? &local_stride : NULL,
                c > 64 ? &local_stride : NULL,
                dtype);
            scale.f32 = 1.f;
            tpu_bdc_fp_avg_pool2d(
                pooling_buffer,
                input_local_addr,
                &pooling_shape,
                &pooling_kernel,
                &pooling_padding,
                &pooling_stride,
                &pooling_dilation,
                dtype,
                scale);
            if(n>1)
            {
                pooling_shape.w = pooling_shape.h;
                pooling_shape.h = 1;
                pooling_kernel.w = pooling_shape.w;
                tpu_bdc_fp_avg_pool2d(
                    c_pooling_buffer,
                    pooling_buffer,
                    &pooling_shape,
                    &pooling_kernel,
                    &pooling_padding,
                    &pooling_stride,
                    &pooling_dilation,
                    dtype,
                    scale);
            }
            tpu_parallel_start();
            tpu_bdc_fp_add(
                batch_var_local_addr,
                batch_var_local_addr,
                c_pooling_buffer,
                &cshape,
                &compact_stride,
                &compact_stride,
                NULL,
                dtype);
            if(hwidx == 2 * hwslice - 1)
            {
                if(tpu_is_parallel_state()){tpu_parallel_end();}
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
                tpu_bdc_fp32_mac_C(
                    running_var_local_addr,
                    batch_var_local_addr,
                    momentum/(n*hw-1.f),
                    &cshape,
                    &compact_stride,
                    &compact_stride);
                tpu_parallel_start();
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
                tpu_bdc_cpy(
                    c_pooling_buffer,
                    batch_var_local_addr,
                    &cshape,
                    NULL,
                    &compact_stride,
                    dtype);
                scale.f32 = 1.f/(n*hw);
                tpu_bdc_fp_scale_bias_C(
                    c_pooling_buffer,
                    c_pooling_buffer,
                    scale,
                    ep,
                    &cshape,
                    dtype);
                tpu_bdc_fp32_rsqrt(
                    c_pooling_buffer,
                    c_pooling_buffer,
                    &cshape);
                tpu_bdc_cpy(
                    batch_var_local_addr,
                    c_pooling_buffer,
                    &cshape,
                    &compact_stride,
                    NULL,
                    dtype);
                if(tpu_is_parallel_state()){tpu_parallel_end();}
                tpu_gdma_compact_L2S(
                    batch_invstd_global_addr,
                    batch_var_local_addr,
                    &cshape,
                    dtype);
            }
        }
        else
        {
            tpu_gdma_cpy_S2L(
                input_local_addr,
                input_global_addr + (hwidx - 2 * hwslice) * hw_secs * tpu_data_type_size(dtype),
                &input_shape,
                NULL,
                &global_stride,
                dtype);
            tpu_bdc_fp_bias(
                input_local_addr,
                input_local_addr,
                batch_mean_local_addr,
                &input_shape,
                dtype);
            tpu_bdc_fp_scale(
                input_local_addr,
                input_local_addr,
                batch_var_local_addr,
                &input_shape,
                dtype);
            tpu_bdc_fp_scale_bias(
                input_local_addr,
                input_local_addr,
                weight_local_addr,
                bias_local_addr,
                &input_shape,
                dtype);
            tpu_gdma_cpy_L2S(
                output_global_addr + (hwidx - 2 * hwslice) * hw_secs * tpu_data_type_size(dtype),
                input_local_addr,
                &input_shape,
                &global_stride,
                NULL,
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
    if(!is_local_mem_enough(shape.n, shape.c, shape.h*shape.w, 1, 0, dtype))
    {
        if(is_local_mem_enough(shape.n, NPU_NUM, shape.h*shape.w, 1, 0, dtype))
        {
            int csecs = ALIGN(shape.c,NPU_NUM)-NPU_NUM;
            while(!is_local_mem_enough(shape.n, csecs, shape.h*shape.w, 1, 0, dtype))
            {
                csecs-=NPU_NUM;
            }
            int cslice = DIV_UP(shape.c, csecs);
            for(int cidx=0; cidx < cslice; cidx++)
            {
                batchnorm_forward_split_c(
                    input_global_addr + cidx * csecs * shape.h * shape.w * tpu_data_type_size(dtype),
                    running_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    running_var_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    weight_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    bias_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    updated_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    updated_var_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    batch_mean_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    batch_invstd_global_addr + cidx * csecs * tpu_data_type_size(dtype),
                    output_global_addr + cidx * csecs * shape.h * shape.w * tpu_data_type_size(dtype),
                    shape,
                    cidx == cslice-1 ? shape.c - cidx * csecs : csecs,
                    momentum,
                    eps,
                    dtype);
            }
        }
        else if(is_local_mem_enough(1, shape.c, shape.h*shape.w, 0, 1, dtype))
        {
            batchnorm_forward_split_n(
                input_global_addr,
                running_mean_global_addr,
                running_var_global_addr,
                weight_global_addr,
                bias_global_addr,
                updated_mean_global_addr,
                updated_var_global_addr,
                batch_mean_global_addr,
                batch_invstd_global_addr,
                output_global_addr,
                shape,
                momentum,
                eps,
                dtype);
        }
        else
        {
            TPUKERNEL_ASSERT(is_local_mem_enough(shape.n, shape.c, tpu_eu_num(dtype), 0, 0, dtype));
            batchnorm_forward_split_hw(
                input_global_addr,
                running_mean_global_addr,
                running_var_global_addr,
                weight_global_addr,
                bias_global_addr,
                updated_mean_global_addr,
                updated_var_global_addr,
                batch_mean_global_addr,
                batch_invstd_global_addr,
                output_global_addr,
                shape,
                momentum,
                eps,
                dtype);
        }
    }
    else
    {
        batchnorm_forward_split_c(
            input_global_addr,
            running_mean_global_addr,
            running_var_global_addr,
            weight_global_addr,
            bias_global_addr,
            updated_mean_global_addr,
            updated_var_global_addr,
            batch_mean_global_addr,
            batch_invstd_global_addr,
            output_global_addr,
            shape,
            shape.c,
            momentum,
            eps,
            dtype);
    }
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
