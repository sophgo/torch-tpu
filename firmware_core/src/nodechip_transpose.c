#include "nodechip_transpose.h"

void tpu_kernel_api_transpose(const void *args)
{
    sg_api_transpose_t *api = (sg_api_transpose_t*)args;
    tpu_initialize();

    int input_shape[8] = {0};
    int order[8] = {0};
    for(int i=0; i<api->dims; ++i)
    {
        input_shape[i] = api->input_shape[i];
        order[i] = api->order[i];
    }
    nodechip_transpose(
        api->input_global_mem_addr,
        api->output_global_mem_addr,
        input_shape,
        order,
        api->dims,
        api->buffer_global_mem_addr,
        NULL,
        tpu_type_convert(api->sgdtype));
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_transpose);