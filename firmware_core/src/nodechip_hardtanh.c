#include "sg_api_struct.h"
#include "tpu_kernel.h"

void nodechip_hardtanh(global_addr_t input_global_addr, global_addr_t output_global_addr,
                       scalar_t min_value, scalar_t max_value, int length, data_type_t dtype) {
    if(length == 0) return;
    if(0 == length) return;
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    const int tensor_num = 2 + 2;     // 2 inputs, 2 outputs
    int tensor_size = LOCAL_MEM_BANKS / tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size > 0);

    const int dtype_size = tpu_data_type_size(dtype);
    local_addr_t input_local_addrs[]  = {0, tensor_size};
    local_addr_t output_local_addrs[] = {2 * tensor_size, 3 * tensor_size};
    int tensor_w = DIV_UP(MIN(length, tensor_size * NPU_NUM / dtype_size), NPU_NUM);

    int todo = length;
    int done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    variable_t min_variable = {.type = SCALAR, .context.scalar = min_value};
    variable_t max_variable = {.type = SCALAR, .context.scalar = max_value};
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
        if (l2s) {
            tpu_gdma_cpy_L2S (l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
        }
        variable_t input_variable = {.type = TENSOR, .context.addr = input_local_addrs[index]};
        variable_t output_variable = {.type = TENSOR, .context.addr = output_local_addrs[index]};
        tpu_bdc_greater_select(output_local_addrs[index], &input_variable, &max_variable,
                               &max_variable, &input_variable, &shape, dtype, dtype);
        tpu_bdc_less_select(output_local_addrs[index], &input_variable, &min_variable,
                            &min_variable, &output_variable, &shape, dtype, dtype);
        l2s = true;
        l2s_global_addr = output_global_addr + done * tpu_data_type_size(dtype);
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
void tpu_kernel_api_hardtanh(const void *args) {
    sg_api_hardtanh_t *api = (sg_api_hardtanh_t*)args;
    scalar_t min_value, max_value;
    if(api->dtype == DT_FP32) {
        min_value.f32 = api->min_value;
        max_value.f32 = api->max_value;
    }
    else {
        scalar_t min_value_f32 = {.f32 = api->min_value};
        scalar_t max_value_f32 = {.f32 = api->max_value};
        min_value = tpu_cast(min_value_f32, (data_type_t)api->dtype, DT_FP32, RM_HALF_TO_EVEN);
        max_value = tpu_cast(max_value_f32, (data_type_t)api->dtype, DT_FP32, RM_HALF_TO_EVEN);
    }

    int length = 1;
    for(int i = 0; i < api->dim; ++i) {
        length *= api->shape[i];
    }
    tpu_initialize();
    nodechip_hardtanh(api->input_global_addr, api->output_global_addr,
                      min_value, max_value, length, api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_hardtanh);

void tpu_kernel_api_hardtanh_multi_core(const void *args) {
    sg_api_hardtanh_t *api = (sg_api_hardtanh_t*)args;
    scalar_t min_value, max_value;
    if(api->dtype == DT_FP32) {
        min_value.f32 = api->min_value;
        max_value.f32 = api->max_value;
    }
    else {
        scalar_t min_value_f32 = {.f32 = api->min_value};
        scalar_t max_value_f32 = {.f32 = api->max_value};
        min_value = tpu_cast(min_value_f32, (data_type_t)api->dtype, DT_FP32, RM_HALF_TO_EVEN);
        max_value = tpu_cast(max_value_f32, (data_type_t)api->dtype, DT_FP32, RM_HALF_TO_EVEN);
    }

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
    nodechip_hardtanh(api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                      min_value, max_value, cur_length_slice, (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_hardtanh_multi_core);