#include "nodechip_softmax.h"
#include "sg_api_struct.h"

void tpu_kernel_api_softmax_forward(const void *args)
{
    sg_api_softmax_forward_t *api = (sg_api_softmax_forward_t *)args;
    tpu_initialize();

    const int shape[] = {api->input_n, api->input_c, api->input_inner_dim};
    nodechip_softmax(
        api->input_global_addr,
        api->output_global_addr,
        shape,
        3,
        1,
        1,
        0,
        api->scale_val,
        tpu_type_convert(api->dtype));
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_softmax_forward);