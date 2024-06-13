#include "sg_api_struct.h"
#include "tpu_kernel.h"


#ifdef BACKEND_SG2260
extern
void nodechip_rmsnorm_forward_multi_core(
    global_addr_t input_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t bias_global_addr,
    global_addr_t output_global_addr,
    int*          shape,
    int           dims,
    int           axis,
    float         partial,
    float         eps,
    int           with_weight,
    int           with_bias,
    data_type_t   dtype);

int tpu_kernel_rmsnorm_multi_core(const void * api_buf)
{
    sg_api_rmsnorm_multi_core_t *api = (sg_api_rmsnorm_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_rmsnorm_forward_multi_core(
        api->input_addr,
        api->weight_addr,
        api->bias_addr,
        api->output_addr,
        api->shape,
        api->dims,
        api->axis,
        api->partial,
        api->eps,
        api->with_weight,
        api->with_bias,
        ( data_type_t ) api->dtype );
    tpu_poll();
    return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_rmsnorm_multi_core);
#endif
