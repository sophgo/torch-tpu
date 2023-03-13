#include "nodechip_binary.h"

void tpu_kernel_api_bcbinary_float(const void *args)
{
    sg_api_bcbinary_float_t *api = (sg_api_bcbinary_float_t*)args;
    tpu_initialize();

    nodechip_bcbinary_fp(
        api->A_global_addr,
        api->B_global_addr,
        api->res_global_addr,
        api->A_shape,
        api->B_shape,
        api->dims,
        api->dims,
        api->binary_type,
        tpu_type_convert(api->dtype),
        0, 0);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_bcbinary_float);
