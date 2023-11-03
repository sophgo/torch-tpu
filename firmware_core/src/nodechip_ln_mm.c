#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

#ifdef FIRMWARE_BACKEND_2260
extern
void nodechip_layernorm_matmul_fuse_multi_core(
    global_addr_t   input_global_addr,
    global_addr_t   gamma_global_addr,
    global_addr_t   beta_global_addr,
    global_addr_t   weight_global_addr,
    global_addr_t   bias_global_addr,
    global_addr_t   norm_mean_global_addr,
    global_addr_t   norm_rstd_global_addr,
    global_addr_t   output_global_addr,
    const int*      in_shape,                 // [batch, M, 1, K]
    const int*      weight_shape,      // [1, K, 1, N]
    const int       in_dims,
    const int       weight_dims,
    data_type_t     dtype,
    float           eps,
    bool            has_bias);


void tpu_kernel_ln_mm_multi_core(const void* api_buf) {

    sg_api_ln_mm_multi_core_t *api = (sg_api_ln_mm_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_layernorm_matmul_fuse_multi_core(
        api->input_addr,
        api->gamma_addr,
        api->beta_addr,
        api->weight_addr,
        api->bias_addr,
        api->mean_addr,
        api->rstd_addr,
        api->output_addr,
        api->in_shape,
        api->w_shape,
        api->in_dims,
        api->w_dims,
        (data_type_t)api->in_dtype,
        api->eps,
        api->has_bias);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_ln_mm_multi_core);
#endif