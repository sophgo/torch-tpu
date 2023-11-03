#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

/*
 * output = pow(input, 1/3)
 */

void nodechip_cbrt(global_addr_t input_global_addr, global_addr_t output_global_addr,
                   int length, data_type_t dtype) {
    if(length==0) return;
    int npu_num=tpu_npu_num();
    int bank_num=tpu_bank_num();
    int bank_size = tpu_local_mem_size_per_npu()/bank_num;
    // 2 inputs, 2 outputs, 2 masks, 2 abs, 2 work0, 2 work1, 1 exp coeff, 1 log coeff, 1 exp table
    int tensor_num=2+2+2+2+2+2+1+1+1;
    int coeff_bank_num=0; // 0 coeff
    int tensor_size = (bank_num-coeff_bank_num)/tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size>0);

    local_addr_t input_local_addrs[2] = {0, 1 * tensor_size};
    local_addr_t output_local_addrs[2] = {2 * tensor_size, 3 * tensor_size};
    local_addr_t mask_local_addrs[2] = {4 * tensor_size, 5 * tensor_size};
    local_addr_t abs_local_addrs[2] = {6 * tensor_size, 7 * tensor_size};
    local_addr_t work1_local_addrs[2] = {8 * tensor_size, 9 * tensor_size};
    local_addr_t work2_local_addrs[2] = {10 * tensor_size, 11 * tensor_size};
    local_addr_t exp_coeff_local_addrs = 12 * tensor_size;
    local_addr_t log_coeff_local_addrs = 13 * tensor_size;
    local_addr_t exp_table_local_addrs = 14 * tensor_size;
    
    int dtype_size = tpu_data_type_size(DT_FP32);
    int tensor_w = DIV_UP(MIN(length, tensor_size*npu_num/dtype_size), npu_num);

    int todo = length;
    int done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    
    tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addrs);
    tpu_bdc_load_fp32_log_coeff(log_coeff_local_addrs);
    tpu_bdc_load_fp32_exp_table(exp_table_local_addrs);
    float scalar = 1./3.;
    scalar_t zero = {.f32 = 0.0}, one = {.f32 = 1.0}, true_value = {.f32 = -2.0};
    while(todo != 0) {
        if(todo > NPU_NUM) {
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, tensor_w);
        } else {
            shape.c = todo;
            shape.w = 1;
        }
        tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dtype_size,
                         &shape, NULL, NULL, dtype);
        if(tpu_is_parallel_state()) {
            tpu_parallel_end();
        }
        tpu_parallel_start();
        if(l2s) {
            tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, DT_FP32);
        }

        if(dtype != DT_FP32) {
            tpu_bdc_cast(input_local_addrs[index], input_local_addrs[index], &shape, NULL,
                         NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);
        }
        tpu_bdc_less_C(abs_local_addrs[index], input_local_addrs[index], zero, true_value,
                       &shape, NULL, NULL, DT_FP32, DT_FP32);
        tpu_bdc_fp_add_C(mask_local_addrs[index], abs_local_addrs[index], one, &shape, NULL, NULL, DT_FP32);  
        tpu_bdc_abs(abs_local_addrs[index], input_local_addrs[index], &shape, NULL, NULL, DT_FP32);
        tpu_bdc_fp32_pow_C(input_local_addrs[index], abs_local_addrs[index], work1_local_addrs[index],
                           work2_local_addrs[index], exp_coeff_local_addrs, log_coeff_local_addrs,
                           exp_table_local_addrs, scalar, &shape);
        tpu_bdc_fp_mul(output_local_addrs[index], input_local_addrs[index], mask_local_addrs[index],
                       &shape, NULL, NULL, NULL, DT_FP32);

        l2s = true;
        l2s_global_addr = output_global_addr + done * dtype_size;
        l2s_local_addr = output_local_addrs[index];
        l2s_shape = shape;
        todo -= shape.c * shape.w;
        done += shape.c * shape.w;
        index = 1 - index;
    }
    if(tpu_is_parallel_state()) {
        tpu_parallel_end();
    }
    if(l2s) {
        tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, DT_FP32);
    }
}

void tpu_kernel_api_cbrt(const void * args) {
    sg_api_cbrt_t *api = (sg_api_cbrt_t*)args;

    int length = 1;
    for(int i = 0; i < api->dim; ++i) {
      length *= api->shape[i];
    }
    tpu_initialize();
    nodechip_cbrt(api->input_global_addr, api->output_global_addr, length, (data_type_t) api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_cbrt);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_cbrt_multi_core(const void *args) {
    sg_api_cbrt_t *api = (sg_api_cbrt_t*)args;
    
    int length = 1;
    for(int i = 0; i < api->dim; ++i) {
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
    nodechip_cbrt(api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                  api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                  cur_length_slice,
                  (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_cbrt_multi_core);
#endif