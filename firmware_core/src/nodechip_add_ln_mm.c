#include "sg_api_struct.h"
#include "tpu_kernel.h"


#ifdef BACKEND_SG2260
extern
void nodechip_mlp0_fuse_multi_core_v3(
    global_addr_t   input0_global_addr,
    global_addr_t   input1_global_addr,
    global_addr_t   gamma_global_addr,
    global_addr_t   beta_global_addr,
    global_addr_t   weight_global_addr,
    global_addr_t   bias_global_addr,
    global_addr_t   norm_out_global_addr,
    global_addr_t   norm_mean_global_addr,
    global_addr_t   norm_rstd_global_addr,
    global_addr_t   output_global_addr,
    const int*      in_shape,                 // [batch, M, 1, K]
    const int*      weight_shape,      // [1, K, 1, N]
    const int       in_dims,
    const int       weight_dims,
    data_type_t     dtype,
    float           eps,
    bool            has_bias,
    bool            use_fast);

void tpu_kernel_add_ln_mm_multi_core(const void* api_buf) {

    sg_api_add_ln_mm_multi_core_t *api = (sg_api_add_ln_mm_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_mlp0_fuse_multi_core_v3(
        api->input0_addr,
        api->input1_addr,
        api->gamma_addr,
        api->beta_addr,
        api->weight_addr,
        api->bias_addr,
        api->out_add_addr,
        api->mean_addr,
        api->rstd_addr,
        api->output_addr,
        api->in_shape,
        api->w_shape,
        api->in_dims,
        api->w_dims,
        (data_type_t)api->in_dtype,
        api->eps,
        api->has_bias,
        api->use_fast);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_add_ln_mm_multi_core);
#endif