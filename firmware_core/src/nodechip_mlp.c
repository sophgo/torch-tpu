#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

#ifdef FIRMWARE_BACKEND_2260
extern
void nodechip_mgm_multi_core(
    global_addr_t   input_gloabl_addr,
    global_addr_t   weight0_gloabl_addr,
    global_addr_t   weight1_gloabl_addr,
    global_addr_t   bias0_gloabl_addr,          // float32
    global_addr_t   bias1_gloabl_addr,
    global_addr_t   output_global_addr,
    const int*      in_shape,
    const int*      weight0_shape,
    const int*      weight1_shape,
    int             in_dims,
    int             weight0_dims,
    int             weight1_dims,
    data_type_t     in_dtype,
    data_type_t     out_dtype,
    int             has_bias,           // has_bias0 + 2 * has_bias1
    bool            use_fast            // gelu or gelu_fast
);

void tpu_kernel_mlp_multi_core(const void* api_buf) {

    sg_api_mlp_multi_core_t *api = (sg_api_mlp_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_mgm_multi_core(
        api->input_addr,
        api->weight0_addr,
        api->weight1_addr,
        api->bias0_addr,
        api->bias1_addr,
        api->output_addr,
        api->in_shape,
        api->w0_shape,
        api->w1_shape,
        api->in_dims,
        api->w0_dims,
        api->w1_dims,
        (data_type_t)api->in_dtype,
        (data_type_t)api->out_dtype,
        api->has_bias,
        api->use_fast);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_mlp_multi_core);
#endif