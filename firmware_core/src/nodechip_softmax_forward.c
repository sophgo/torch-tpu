#include "nodechip_softmax.h"
#include "sg_api_struct.h"

void tpu_kernel_api_softmax_forward(const void *args)
{
    sg_api_softmax_forward_t *api = (sg_api_softmax_forward_t *)args;
    tpu_initialize();

    nodechip_softmax(
        api->input_global_addr,
        api->output_global_addr,
        api->shape,
        api->dims,
        api->compute_dim,
        api->compute_dim,
        0,
        api->scale_val,
        tpu_type_convert(api->dtype));
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_softmax_forward);
