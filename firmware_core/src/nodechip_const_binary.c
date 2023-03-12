#include "nodechip_binary.h"

void tpu_kernel_api_const_binary(const void *args)
{
    sg_api_const_binary_float_t *api = (sg_api_const_binary_float_t*)args;
    tpu_initialize();

    nodechip_const_binary_fp(
        api->input_addr,
        api->output_addr,
        api->shape,
        api->dims,
        api->const_value,
        api->is_inversed,
        api->binary_type,
        tpu_type_convert(api->dtype),
        0, 0);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_const_binary);
