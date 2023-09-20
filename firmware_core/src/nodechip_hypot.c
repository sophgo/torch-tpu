#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = sqrt(pow(input, 2) + pow(other, 2))
 */

void nodechip_hypot(
global_addr_t input_global_addr,
global_addr_t other_global_addr,
global_addr_t output_global_addr,
int length,
data_type_t dtype) {
    if(0 == length) return;
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    // // 2 inputs, 2 others, 2 outputs, 2 pow, 2 sum, 1 work1, 1 work2, 1 exp_coeff, 1 log, 1 exp_table
    // const int tensor_num = 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1;
    const int tensor_num = 2 + 2 + 2 + 2 + 2;   // 2 inputs, 2 others, 2 outputs, 2 pow, 2 sum
    int tensor_size = LOCAL_MEM_BANKS / tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size > 0);

    const int dtype_size = tpu_data_type_size(dtype);
    local_addr_t input_local_addrs[] = {0, tensor_size};
    local_addr_t other_local_addrs[] = {2 * tensor_size, 3 * tensor_size};
    local_addr_t output_local_addrs[] = {4 * tensor_size, 5 * tensor_size};
    local_addr_t pow_local_addrs[] = {6 * tensor_size, 7 * tensor_size};
    local_addr_t sum_local_addrs[] = {8 * tensor_size, 9 * tensor_size};
    // local_addr_t work1_local_addr = 10 * tensor_size;
    // local_addr_t work2_local_addr = 11 * tensor_size;
    // local_addr_t exp_coeff_local_addr = 12 * tensor_size;
    // local_addr_t log_local_addr = 13 * tensor_size;
    // local_addr_t exp_table_local_addr = 14 * tensor_size;
    int tensor_w = DIV_UP(MIN(length, tensor_size * NPU_NUM / dtype_size), NPU_NUM);

    // tpu_bdc_load_fp32_exp_coeff(exp_coeff_local_addr);
    // tpu_bdc_load_fp32_log_coeff(log_local_addr);
    // tpu_bdc_load_fp32_exp_table(exp_table_local_addr);

    int todo = length, done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    while(todo != 0) {
        if(todo > NPU_NUM){
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, tensor_w);
        }
        else {
            shape.c = todo;
            shape.w = 1;
        }

        tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dtype_size,
                         &shape, NULL, NULL, dtype);
        tpu_gdma_cpy_S2L(other_local_addrs[index], other_global_addr + done * dtype_size,
                         &shape, NULL, NULL, dtype);

        if(tpu_is_parallel_state()) {
            tpu_parallel_end();
        }
        tpu_parallel_start();

        if(l2s) {
            tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
        }

        tpu_bdc_fp_mul(pow_local_addrs[index], input_local_addrs[index],
                       input_local_addrs[index], &shape, NULL, NULL, NULL, dtype);
        tpu_bdc_fp_mul(output_local_addrs[index], other_local_addrs[index],
                       other_local_addrs[index], &shape, NULL, NULL, NULL, dtype);
        tpu_bdc_fp_add(sum_local_addrs[index], pow_local_addrs[index],
                       output_local_addrs[index], &shape, NULL, NULL, NULL, dtype);
        tpu_bdc_fp32_sqrt(output_local_addrs[index], sum_local_addrs[index], &shape);

        // tpu_bdc_fp32_pow_C(pow_local_addrs[index], input_local_addrs[index],
        //                    work1_local_addr, work2_local_addr, exp_coeff_local_addr,
        //                    log_local_addr, exp_table_local_addr, 2., &shape);
        // tpu_bdc_fp32_pow_C(output_local_addrs[index], other_local_addrs[index],
        //                    work1_local_addr, work2_local_addr, exp_coeff_local_addr,
        //                    log_local_addr, exp_table_local_addr, 2., &shape);
        // tpu_bdc_fp_add(sum_local_addrs[index], output_local_addrs[index],
        //                pow_local_addrs[index], &shape, NULL, NULL, NULL, dtype);
        // tpu_bdc_fp32_tunable_sqrt(output_local_addrs[index], sum_local_addrs[index], &shape, 4);                 

        l2s = true;
        l2s_global_addr = output_global_addr + done * dtype_size;
        l2s_local_addr = output_local_addrs[index];
        l2s_shape = shape;
        todo -= shape.c * shape.w;
        done += shape.c * shape.w;
        index = 1- index;
    }
    if(tpu_is_parallel_state()) {
        tpu_parallel_end();
    }
    if(l2s) {
        tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
    }
}
void tpu_kernel_api_hypot(const void *args) {
    sg_api_hypot_t *api = (sg_api_hypot_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32);

    int length = 1;
    for(int i = 0; i < api->dim; ++i){
        length *= api->shape[i];
    }
    tpu_initialize();
    nodechip_hypot(api->input_global_addr, api->other_global_addr,
                   api->output_global_addr, length, (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_hypot);

void nodechip_hypot_bcast(
global_addr_t input_global_addr,
global_addr_t other_global_addr,
global_addr_t output_global_addr,
const dim4   *input_shape,
const dim4   *other_shape,
const dim4   *output_shape,
data_type_t   dtype) {
    dim4 input_global_stride, other_global_stride, output_global_stride;
    tpu_continuous_stride(&input_global_stride,  input_shape);
    tpu_continuous_stride(&other_global_stride,  other_shape);
    tpu_continuous_stride(&output_global_stride, output_shape);

    const bool input_bcast[4] = {
        input_shape->n != output_shape->n,
        input_shape->c != output_shape->c,
        input_shape->h != output_shape->h,
        input_shape->w != output_shape->w
    };
    const bool other_bcast[4] = {
        other_shape->n != output_shape->n,
        other_shape->c != output_shape->c,
        other_shape->h != output_shape->h,
        other_shape->w != output_shape->w
    };
    bool input_bcast_all = false, other_bcast_all = false;
    for(int i = 0; i < 4; ++i) {
        input_bcast_all = input_bcast_all || input_bcast[i];
        other_bcast_all = other_bcast_all || other_bcast[i];
    }

    const int c_per_npu = DIV_UP ( output_shape->c, NPU_NUM );
    int hmax = output_shape->h, nmax = output_shape->n, cmax = c_per_npu * NPU_NUM;
    local_addr_t output_addr, input_addr, other_addr, work1_addr, work2_addr;
    while(true) {
        output_addr = 0;
        int output_size = tpu_aligned_feature_size(hmax, output_shape->w, dtype) * DIV_UP(cmax, NPU_NUM) * nmax;
        input_addr = output_addr + output_size;
        int input_size = output_size;
        other_addr = input_addr + input_size;
        int other_size = output_size;
        work1_addr = other_addr + other_size;
        int work1_size = output_size;
        work2_addr = work1_addr + work1_size;
        int work2_size = output_size;
        int total_size = work2_addr + work2_size;
        if(total_size <= LOCAL_MEM_SIZE) {
            break;
        }
        else {
            if(cmax > NPU_NUM) {
                if(cmax % NPU_NUM == 0) {
                    cmax -= NPU_NUM;
                }
                else {
                    cmax -= (cmax % NPU_NUM);
                }
                continue;
            }
            else if(nmax > 1) {
                nmax /= 2;
                continue;
            }
            else if(hmax > 1) {
                hmax /= 2;
                continue;
            }
            else {
                TPUKERNEL_ASSERT ( false );
            }
        }
    }
    dim4 shape = { .w = output_shape->w };
    dim4 input_local_shape, other_local_shape;
    dim4 input_local_stride, other_local_stride;
    int ctodo = output_shape->c, cdone = 0;
    while(ctodo > 0) {
        shape.c = MIN(ctodo, cmax);
        int ntodo = output_shape->n, ndone = 0;
        while(ntodo > 0) {
            shape.n = MIN(ntodo, nmax);
            int htodo = output_shape->h, hdone = 0;
            while(htodo > 0) {
                shape.h = MIN (htodo, hmax);
                // Move input from global memory to local memory
                tpu_aligned_stride (&input_local_stride, 0, &shape, dtype);
                input_local_shape.n = input_bcast[0] ? 1 : shape.n;
                input_local_shape.c = input_bcast[1] ? 1 : shape.c;
                input_local_shape.h = input_bcast[2] ? 1 : shape.h;
                input_local_shape.w = input_bcast[3] ? 1 : shape.w;
                global_addr_t input_global_addr_gdma = input_global_addr +
                                ((input_bcast[0] ? 0 : ndone) * input_global_stride.n +
                                 (input_bcast[1] ? 0 : cdone) * input_global_stride.c +
                                 (input_bcast[2] ? 0 : hdone) * input_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_S2L(input_addr, input_global_addr_gdma, &input_local_shape,
                                 &input_local_stride, &input_global_stride, dtype);
                
                // Move other from global memory to local memory
                tpu_aligned_stride(&other_local_stride, 0, &shape, dtype);
                other_local_shape.n = other_bcast[0] ? 1 : shape.n;
                other_local_shape.c = other_bcast[1] ? 1 : shape.c;
                other_local_shape.h = other_bcast[2] ? 1 : shape.h;
                other_local_shape.w = other_bcast[3] ? 1 : shape.w;
                global_addr_t other_global_addr_gdma = other_global_addr +
                                ((other_bcast[0] ? 0 : ndone) * other_global_stride.n +
                                 (other_bcast[1] ? 0 : cdone) * other_global_stride.c +
                                 (other_bcast[2] ? 0 : hdone) * other_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_S2L(other_addr, other_global_addr_gdma, &other_local_shape,
                                 &other_local_stride, &other_global_stride, dtype);
                
                // Broadcast input if needed
                if(input_bcast[1]) {
                    input_local_shape.c = NPU_NUM;
                    tpu_bdc_npu_bcast(input_addr, input_addr, &input_local_shape, dtype);
                }
                if(input_bcast[0] || input_bcast[2] || input_bcast[3] || (input_bcast[1] && shape.c > NPU_NUM)) {
                    dim4 input_bcast_stride;
                    input_bcast_stride.n = input_bcast[0] ? 0 : input_local_stride.n;
                    input_bcast_stride.c = input_bcast[1] ? 0 : input_local_stride.c;
                    input_bcast_stride.h = input_bcast[2] ? 0 : input_local_stride.h;
                    input_bcast_stride.w = input_bcast[3] ? 0 : input_local_stride.w;
                    tpu_bdc_cpy(input_addr, input_addr, &shape, NULL, &input_bcast_stride, dtype);
                }
                
                // Broadcast other if needed
                if(other_bcast[1]) {
                    other_local_shape.c = NPU_NUM;
                    tpu_bdc_npu_bcast(other_addr, other_addr, &other_local_shape, dtype);
                }
                if(other_bcast[0] || other_bcast[2] || other_bcast[3] || (other_bcast[1] && shape.c > NPU_NUM)) {
                    dim4 other_bcast_stride;
                    other_bcast_stride.n = other_bcast[0] ? 0 : other_local_stride.n;
                    other_bcast_stride.c = other_bcast[1] ? 0 : other_local_stride.c;
                    other_bcast_stride.h = other_bcast[2] ? 0 : other_local_stride.h;
                    other_bcast_stride.w = other_bcast[3] ? 0 : other_local_stride.w;
                    tpu_bdc_cpy ( other_addr, other_addr, &shape, NULL, &other_bcast_stride, dtype );
                }
                
                // Select
                tpu_bdc_fp_mul(output_addr, input_addr, input_addr, &shape, NULL, NULL, NULL, dtype);
                tpu_bdc_fp_mul(work1_addr, other_addr, other_addr, &shape, NULL, NULL, NULL, dtype);
                tpu_bdc_fp_add(work2_addr, output_addr, work1_addr, &shape, NULL, NULL, NULL, dtype);
                tpu_bdc_fp32_sqrt(output_addr, work2_addr, &shape);
                
                // Move out from local memory to global memory
                global_addr_t output_global_addr_gdma = 
                                            output_global_addr +
                                            (ndone * output_global_stride.n +
                                             cdone * output_global_stride.c +
                                             hdone * output_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_L2S(output_global_addr_gdma, output_addr, &shape,
                                 &output_global_stride, NULL, dtype);
                htodo -= shape.h;
                hdone += shape.h;
            }
            ntodo -= shape.n;
            ndone += shape.n;
        }
        ctodo -= shape.c;
        cdone += shape.c;
    }
}
void tpu_kernel_api_hypot_bcast(const void *args) {
    sg_api_hypot_bcast_t *api = (sg_api_hypot_bcast_t*) args;
    TPUKERNEL_ASSERT(api->dim > 0 && api->dim <= 4);
    TPUKERNEL_ASSERT(api->dtype == DT_FP32);

    dim4 input_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 other_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    if(api->dim >= 1) {
        input_shape.n = api->input_shape[0];
        other_shape.n = api->other_shape[0];
        output_shape.n = input_shape.n > other_shape.n ? input_shape.n : other_shape.n;
    }
    if(api->dim >= 2) {
        input_shape.c = api->input_shape[1];
        other_shape.c = api->other_shape[1];
        output_shape.c = input_shape.c > other_shape.c ? input_shape.c : other_shape.c;
    }
    if(api->dim >= 3) {
        input_shape.h = api->input_shape[2];
        other_shape.h = api->other_shape[2];
        output_shape.h = input_shape.h > other_shape.h ? input_shape.h : other_shape.h;
    }
    if(api->dim >= 4) {
        input_shape.w = api->input_shape[3];
        other_shape.w = api->other_shape[3];
        output_shape.w = input_shape.w > other_shape.w ? input_shape.w : other_shape.w;
    }

    tpu_initialize();
    nodechip_hypot_bcast(api->input_global_addr,
                         api->other_global_addr,
                         api->output_global_addr,
                         &input_shape,
                         &other_shape,
                         &output_shape,
                         api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_hypot_bcast);

void nodechip_hypot_c(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
float scalar,
int length,
data_type_t dtype) {
    if(0 == length) return;
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    const int tensor_num = 2 + 2 + 2; // 2 inputs, 2 outputs, 2 work
    int tensor_size = LOCAL_MEM_BANKS / tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size > 0);

    const int dtype_size = tpu_data_type_size(dtype);
    local_addr_t input_local_addrs[]  = {0, tensor_size};
    local_addr_t output_local_addrs[] = {2 * tensor_size, 3 * tensor_size};
    local_addr_t work_local_addrs[] = {4 * tensor_size, 5 * tensor_size};
    int tensor_w = DIV_UP(MIN(length, tensor_size * NPU_NUM / dtype_size), NPU_NUM);

    int todo = length;
    int done = 0;
    dim4 shape = { .n = 1, .h = 1 };
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    scalar_t value = {.f32 = scalar * scalar};
    while(todo != 0) {
        if(todo > NPU_NUM) {
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, tensor_w);
        }
        else {
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
            tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
        }

        tpu_bdc_fp_mul(output_local_addrs[index], input_local_addrs[index],
                       input_local_addrs[index], &shape, NULL, NULL, NULL, dtype);
        tpu_bdc_fp_add_C(work_local_addrs[index], output_local_addrs[index],
                         value, &shape, NULL, NULL, dtype);
        tpu_bdc_fp32_sqrt(output_local_addrs[index], work_local_addrs[index], &shape);

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
        tpu_gdma_cpy_L2S(l2s_global_addr,
                         l2s_local_addr,
                         &l2s_shape,
                         NULL,
                         NULL,
                         dtype);
    }
}
void tpu_kernel_api_hypot_c(const void *args) {
    sg_api_hypot_c_t *api = (sg_api_hypot_c_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32);

    int length = 1;
    for(int i = 0; i < api->dim; ++i) {
        length *= api->shape[i];
    }

    tpu_initialize();
    nodechip_hypot_c(api->input_global_addr,
                     api->output_global_addr,
                     api->scalar,
                     length,
                     (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_hypot_c);
