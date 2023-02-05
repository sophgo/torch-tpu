#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

/**
 * linear / fc layer
 * forward：
 *      y = x @ weight` + bias
 *      x (batch, if)
 *      weight (of, if)
 *      bias (of)
 *      y (batch, of)
 * backward:
 *      dx = dy @ weight
 *      dw = dy` @ x
 *      db = Σdy
*/

static inline bool is_local_mem_enough(
    int n,
    int in_features,
    int out_features,
    data_type_t dtype)
{
    int input_size = DIV_UP(n, NPU_NUM) * tpu_aligned_feature_size(in_features, 1, dtype);
    int output_size = DIV_UP(n, NPU_NUM) * tpu_aligned_feature_size(out_features, 1, dtype);
    int weight_size = DIV_UP(out_features, NPU_NUM) * tpu_aligned_feature_size(in_features, 1, dtype);
    int output_trans_size = DIV_UP(out_features, NPU_NUM) * tpu_aligned_feature_size(n, 1, dtype);
    int total_size = ALIGN(input_size, BANK_SIZE) + 
                     ALIGN(output_size, BANK_SIZE) +
                     ALIGN(output_trans_size, BANK_SIZE) + 
                     ALIGN(weight_size, BANK_SIZE) * 2;
    return total_size < LOCAL_MEM_SIZE;
}

void linear_backward(
    global_addr_t input_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t grad_output_global_addr,
    global_addr_t grad_input_global_addr,
    global_addr_t grad_weight_global_addr,
    global_addr_t grad_bias_global_addr,
    int batch,
    int in_features,
    int out_features,
    int grad_weight_enable,
    int grad_bias_enable,
    data_type_t dtype)
{
    dim4 input_shape = {1, batch, 1, in_features};
    dim4 output_shape = {1, batch, 1, out_features};
    dim4 trans_shape = {1, out_features, 1, batch};
    dim4 weight_shape = {1, out_features, 1, in_features};
    dim4 bias_shape = {1, out_features, 1, 1};
    int input_size = DIV_UP(batch, NPU_NUM) * tpu_aligned_feature_size(in_features, 1, dtype);
    int output_size = DIV_UP(batch, NPU_NUM) * tpu_aligned_feature_size(out_features, 1, dtype);
    int weight_size = DIV_UP(out_features, NPU_NUM) * tpu_aligned_feature_size(in_features, 1, dtype);
    int output_trans_size = DIV_UP(out_features, NPU_NUM) * tpu_aligned_feature_size(batch, 1, dtype);
    local_addr_t grad_output_local_addr = 0;
    local_addr_t input_local_addr = grad_output_local_addr + ALIGN(output_size, BANK_SIZE);
    local_addr_t grad_output_trans_addr = input_local_addr + ALIGN(input_size, BANK_SIZE);
    local_addr_t weight_local_addr = grad_output_trans_addr + ALIGN(output_trans_size, BANK_SIZE);
    local_addr_t grad_weight_local_addr = weight_local_addr + ALIGN(weight_size, BANK_SIZE);
    local_addr_t grad_input_local_addr = input_local_addr;
    local_addr_t grad_bias_local_addr = weight_local_addr;
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &output_shape,
        NULL,
        NULL,
        dtype);
    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        input_local_addr,
        input_global_addr,
        &input_shape,
        NULL,
        NULL,
        dtype);
    // tpu_gdma_cpy_cw_trans_L2L(
    //     grad_output_trans_addr,
    //     grad_output_local_addr,
    //     &trans_shape,
    //     NULL,
    //     NULL,
    //     dtype);
    tpu_bdc_wc_trans(
        grad_output_trans_addr,
        grad_output_local_addr,
        &trans_shape,
        dtype);
    tpu_parallel_end();
    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        weight_local_addr,
        weight_global_addr,
        &weight_shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_mm(
        grad_weight_local_addr,
        grad_output_trans_addr,
        input_local_addr,
        out_features,
        batch,
        in_features,
        dtype,
        dtype,
        false);
    tpu_parallel_end();
    if(grad_weight_enable)
    {
        tpu_parallel_start();
        tpu_gdma_cpy_L2S(
            grad_weight_global_addr,
            grad_weight_local_addr,
            &weight_shape,
            NULL,
            NULL,
            dtype);
    }
    tpu_bdc_fp_mm(
        grad_input_local_addr,
        grad_output_local_addr,
        weight_local_addr,
        batch,
        out_features,
        in_features,
        dtype,
        dtype,
        false);
    if(tpu_is_parallel_state())tpu_parallel_end();
    tpu_parallel_start();
    tpu_gdma_cpy_L2S(
        grad_input_global_addr,
        grad_input_local_addr,
        &input_shape,
        NULL,
        NULL,
        dtype);
    scalar_t scale = {.f32 = 1.};
    dim2 pooling_kernel = {1, trans_shape.w};
    dim2 pooling_stride = {1, 1};
    dim2 pooling_dilation = {1, 1};
    padding_t pooling_padding = {0, 0, 0, 0};
    tpu_bdc_fp_avg_pool2d(
        grad_bias_local_addr,
        grad_output_trans_addr,
        &trans_shape,
        &pooling_kernel,
        &pooling_padding,
        &pooling_stride,
        &pooling_dilation,
        dtype,
        tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
    tpu_parallel_end();
    if(grad_bias_enable)
    {
        tpu_gdma_cpy_L2S(
            grad_bias_global_addr,
            grad_bias_local_addr,
            &bias_shape,
            NULL,
            NULL,
            dtype);
    }
}

void nodechip_linear_backward(
    global_addr_t input_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t grad_output_global_addr,
    global_addr_t grad_input_global_addr,
    global_addr_t grad_weight_global_addr,
    global_addr_t grad_bias_global_addr,
    int batch,
    int inf,
    int of,
    int grad_input_enable,
    int grad_weight_enable,
    int grad_bias_enable,
    data_type_t dtype)
{
    TPUKERNEL_ASSERT(is_local_mem_enough(1, inf, of, dtype));
    if(!is_local_mem_enough(batch, inf, of, dtype))
    {
        if(is_local_mem_enough(NPU_NUM, inf, of, dtype))
        {
            int secs = batch;
            while(!is_local_mem_enough(batch, inf, of, dtype))
            {
                secs--;
            }
            int slice = DIV_UP(batch, secs);
            for(int idx=0; idx < slice; ++idx)
            {
                linear_backward(
                    input_global_addr + idx * secs * tpu_data_type_size(dtype),
                    weight_global_addr,
                    grad_output_global_addr + idx * secs * tpu_data_type_size(dtype),
                    grad_input_global_addr + idx * secs * tpu_data_type_size(dtype),
                    grad_weight_global_addr,
                    grad_bias_global_addr,
                    secs,
                    inf,
                    of,
                    grad_weight_enable,
                    grad_bias_enable,
                    dtype);
            }
        }
    }
    else
    {
        linear_backward(
            input_global_addr,
            weight_global_addr,
            grad_output_global_addr,
            grad_input_global_addr,
            grad_weight_global_addr,
            grad_bias_global_addr,
            batch,
            inf,
            of,
            grad_weight_enable,
            grad_bias_enable,
            dtype);
    }
}

void tpu_kernel_api_linear_backward(const void *args)
{
    sg_api_linear_backward_t *api = (sg_api_linear_backward_t *)args;

    tpu_initialize();
    nodechip_linear_backward(
            api->input_global_addr,
            api->weight_global_addr,
            api->grad_output_global_addr,
            api->grad_input_global_addr,
            api->grad_weight_global_addr,
            api->grad_bias_global_addr,
            api->batch,
            api->features[0],
            api->features[1],
            api->grad_input_enable,
            api->grad_weight_enable,
            api->grad_bias_enable,
            DT_FP16);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_linear_backward);
