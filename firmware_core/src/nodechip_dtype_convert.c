#include "nodechip_cast.h"
#include "sg_api_struct.h"

void tpu_kernel_api_dtype_convert(const void* args) {
    sg_api_dtype_convert_t* api = (sg_api_dtype_convert_t*)args;

    int shape[4] = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};

    tpu_initialize();
    nodechip_cast(
        api->input_global_addr,
        api->output_global_addr,
        shape,
        api->dims,
        tpu_type_convert((sg_data_type_t)api->idtype),
        tpu_type_convert((sg_data_type_t)api->odtype),
        tpu_round_mode_convert((sg_round_mode_t)api->round_mode));
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_dtype_convert);
