#include "nodechip_relu_parallel.h"
#include "sg_api_struct.h"

void tpu_kernel_api_relu_forward(const void *args)
{
    sg_api_relu_forward_t *api = (sg_api_relu_forward_t *)args;
    // data_type_t dtype = DT_FP16;

    tpu_initialize();
    nodechip_relu_parallel(
            api->input_global_addr,
            api->output_global_addr,
            api->upper_limit,
            api->shape[0],
            api->shape[1],
            api->shape[2],
            api->shape[3],
            api->dtype?DT_FP16:DT_FP32);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_relu_forward);