#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern void nodechip_masked_fill(
    global_addr_t  input_global_addr,
    global_addr_t  mask_global_addr,
    global_addr_t  output_global_addr,
    const int*     input_shape,
    const int*     mask_shape,
    int            input_dims,
    int            mask_dims,
    float          value,
    data_type_t    dtype);


void tpu_kernel_api_masked_fill ( const void * args )
{
    sg_api_masked_fill_t *api = ( sg_api_masked_fill_t * ) args;
    tpu_initialize();
    nodechip_masked_fill(
        api->input_global_addr,
        api->mask_global_addr,
        api->out_global_addr,
        api->input_shape,
        api->mask_shape,
        api->input_dims,
        api->mask_dims,
        api->value,
        api->dtype
    );
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_masked_fill);