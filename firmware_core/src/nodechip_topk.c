#include <stdlib.h>

#include "sg_api_struct.h"
#include "tpu_kernel.h"

// extern void nodechip_transpose(
// global_addr_t         input_global_addr,
// global_addr_t         output_global_addr,
// int*                  input_shape,
// int*                  order,
// int                   dims,
// global_addr_t         buffer_global_addr,
// unsigned long long*   buffer_size,
// data_type_t           dtype);

void nodechip_topk(
global_addr_t input_global_addr,
global_addr_t value_global_addr,
global_addr_t index_global_addr,
global_addr_t index_buffer_global_addr,
global_addr_t trans_buffer_global_addr,
int *shape,
int dim,
int k,
int dim_order,
bool largest,
bool sorted,
int64_t length,
data_type_t dtype) {
    if(length == 0) return;
    if(dim_order < 0) dim_order += dim;

    // transpose to w
    if(dim_order != dim - 1) {
        // need todo ?
    }

    const int batch_stride = shape[dim_order];
    const int batch = length / batch_stride;
    int *batch_nums = (int*)malloc(batch * sizeof(int));
    for(int i = 0 ; i < batch; ++i) {
        batch_nums[i] = batch_stride;
    }
    // topk for w
    global_addr_t src_value_addr = input_global_addr;
    global_addr_t dst_value_addr = value_global_addr;
    global_addr_t buffer_index_addr = index_buffer_global_addr;
    global_addr_t dst_index_addr = index_global_addr;
    dim4 index_shape = {.n = 1, .c = 1, .h = 1};
    dim4 index_stride = {.w = 2};
    for(int i = 0; i < batch; ++i) {
        src_value_addr = input_global_addr + i * batch_stride * sizeof(int);
        dst_value_addr = value_global_addr + i * k * sizeof(int);
        buffer_index_addr = index_buffer_global_addr + i * k * sizeof(int);
        tpu_hau_sort_natural_index(dst_value_addr, buffer_index_addr, src_value_addr,
                                   batch_nums[i], k, largest, dtype);
        index_shape.w = batch_nums[i];
        index_stride.n = index_stride.c = index_stride.h = batch_nums[i] * 2;
        tpu_gdma_cpy_S2S(dst_index_addr + i * k * sizeof(int) * 2, buffer_index_addr,
                         &index_shape, &index_stride, NULL, DT_INT32);
        tpu_hau_poll();
    }
}
void tpu_kernel_api_topk(const void *args) {
    sg_api_topk_t *api = (sg_api_topk_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_INT32 || api->dtype == DT_UINT32);

    int64_t length = 1;
    for(int i = 0 ; i < api->dim; ++i) {
        length *= api->shape[i];
    }

    tpu_initialize();
    nodechip_topk(api->input_global_addr, api->value_global_addr, api->index_global_addr,
                  api->index_buffer_global_addr, api->trans_buffer_global_addr, api->shape,
                  api->dim, api->k, api->dim_order, api->largest, api->sorted, length, api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_topk);