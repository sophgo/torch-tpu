#include "nodechip_active.h"
#include "nodechip_active_local.h"
#include "sg_api_struct.h"

void tpu_kernel_api_active_forward(const void *args)
{
    sg_api_active_forward_t *api = (sg_api_active_forward_t*)args;
    tpu_initialize();

    nodechip_active(
        api->in_global_addr,
        api->out_global_addr,
        api->shape,
        api->shape_dim,
        api->dtype,
        api->active_type,
        NULL);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_active_forward);
