#include "nodechip_fc.h"
#include "sg_api_struct.h"

void tpu_kernel_api_linear_forward(const void *args)
{
    sg_api_linear_forward_t *api = (sg_api_linear_forward_t *)args;
    data_type_t dtype = DT_FP16;

    tpu_initialize();
    nodechip_fc(
            api->input_global_addr,
            api->weight_global_addr,
            api->bias_global_addr,
            api->output_global_addr,
            api->batch,
            api->in_features,
            api->out_features,
            1,
            1,
            dtype,
            dtype,
            dtype,
            0,
            0);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_linear_forward);