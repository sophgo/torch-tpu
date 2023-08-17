#include "sg_api_struct.h"
#include "tpu_kernel.h"

void nodechip_equal(
global_addr_t input_global_addr,
global_addr_t other_global_addr,
global_addr_t output_global_addr,
int length,
data_type_t dtype) {
    const int dsize = tpu_data_type_size(dtype);
    int wmax = DIV_UP(length, NPU_NUM);
    local_addr_t input_local_addrs[2], other_local_addrs[2], output_local_addrs[2];
    local_addr_t next = 0;
    int size = tpu_aligned_feature_size(1, wmax, dtype);
    while(true) {
        next = 0;
        input_local_addrs[0]  = next; next += size;
        input_local_addrs[1]  = next; next += size;
        other_local_addrs[0]  = next; next += size;
        other_local_addrs[1]  = next; next += size;
        output_local_addrs[0] = next; next += size;
        output_local_addrs[1] = next; next += size;
        if((int)next <= LOCAL_MEM_SIZE) {
            break;
        }
        else {
            if(wmax > 1) {
                wmax /= 2;
                continue;
            }
            else {
                TPUKERNEL_ASSERT(false);
            }
        }
    }

    int todo = length, done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    scalar_t one_u8 = {.u8 = 1};
    while(todo != 0) {
        if(todo > NPU_NUM){
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, wmax);
        }
        else {
            shape.c = todo;
            shape.w = 1;
        }
        tpu_gdma_cpy_S2L(input_local_addrs[index],
                         input_global_addr + done * dsize,
                         &shape,
                         NULL,
                         NULL,
                         dtype);
        tpu_gdma_cpy_S2L(other_local_addrs[index],
                         other_global_addr + done * dsize,
                         &shape,
                         NULL,
                         NULL,
                         dtype);
        if(tpu_is_parallel_state()) {
            tpu_parallel_end();
        }
        tpu_parallel_start();
        if(l2s) {
            tpu_gdma_cpy_L2S(l2s_global_addr,
                             l2s_local_addr,
                             &l2s_shape,
                             NULL,
                             NULL,
                             DT_UINT8);
        }
        tpu_bdc_equal(output_local_addrs[index],
                      input_local_addrs[index],
                      other_local_addrs[index],
                      one_u8,
                      &shape,
                      NULL,
                      NULL,
                      NULL,
                      DT_UINT8,
                      dtype);
        l2s = true;
        l2s_global_addr = output_global_addr + done * dsize;
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
        tpu_gdma_cpy_L2S(l2s_global_addr,
                         l2s_local_addr,
                         &l2s_shape,
                         NULL,
                         NULL,
                         DT_UINT8);
    }
}
void tpu_kernel_api_equal(const void *args) {
    sg_api_equal_t *api = (sg_api_equal_t*) args;
    int length = 1;
    for(int i = 0; i < api->dim; ++i) {
        length *= api->shape[i];
    }

    tpu_initialize();

    nodechip_equal(api->input_global_addr,
                   api->other_global_addr,
                   api->output_global_addr,
                   length,
                   (data_type_t)api->dtype);

    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_equal);

void nodechip_equal_c(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
scalar_t value,
int length,
data_type_t dtype) {
    const int dsize = tpu_data_type_size(dtype);
    int wmax = DIV_UP(length, NPU_NUM);
    local_addr_t input_local_addrs[2], output_local_addrs[2];
    local_addr_t next;
    while(true) {
        next = 0;
        int size = tpu_aligned_feature_size(1, wmax, dtype);
        input_local_addrs[0]  = next; next += size;
        input_local_addrs[1]  = next; next += size;
        output_local_addrs[0] = next; next += size;
        output_local_addrs[1] = next; next += size;
        if((int)next <= LOCAL_MEM_SIZE) {
            break;
        }
        else {
            if(wmax > 1) {
                wmax /= 2;
                continue;
            }
            else {
                TPUKERNEL_ASSERT(false);
            }
        }
    }

    int todo = length;
    int done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    while(todo != 0) {
        if(todo > NPU_NUM) {
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, wmax);
        }
        else {
            shape.c = todo;
            shape.w = 1;
        }
        tpu_gdma_cpy_S2L(input_local_addrs[index],
                         input_global_addr + done * dsize,
                         &shape,
                         NULL,
                         NULL,
                         dtype);
        if(tpu_is_parallel_state()) {
            tpu_parallel_end();
        }
        tpu_parallel_start();
        if (l2s) {
        tpu_gdma_cpy_L2S (l2s_global_addr,
                          l2s_local_addr,
                          &l2s_shape,
                          NULL,
                          NULL,
                          DT_UINT8);
        }
        scalar_t one_u8 = {.u8 = 1};
        tpu_bdc_equal_C(output_local_addrs[index],
                        input_local_addrs[index],
                        value,
                        one_u8,
                        &shape,
                        NULL,
                        NULL,
                        DT_UINT8,
                        dtype);
        l2s = true;
        l2s_global_addr = output_global_addr + done * dsize;
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
                         DT_UINT8);
    }
}
void tpu_kernel_api_equal_c(const void *args) {
    sg_api_equal_c_t *api = (sg_api_equal_c_t*)args;
    scalar_t value;
    if(api->dtype == DT_FP32) {
        value.f32 = api->const_value;
    }
    else {
        scalar_t value_f32 = {.f32 = api->const_value};
        value = tpu_cast(value_f32, (data_type_t)api->dtype, DT_FP32, RM_HALF_TO_EVEN);
    }

    int length = 1;
    for(int i = 0; i < api->dim; ++i) {
        length *= api->shape[i];
    }
    tpu_initialize();

    nodechip_equal_c(api->input_global_addr,
                     api->output_global_addr,
                     value,
                     length,
                     api->dtype);

    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_equal_c);