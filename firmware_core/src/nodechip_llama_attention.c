#include "sg_api_struct.h"
#include "tpu_kernel.h"


#ifdef BACKEND_SG2260
// #ifdef 1
extern
void nodechip_llama2_qkv_multi_core(
    global_addr_t   Q_global_addr,
    global_addr_t   K_global_addr,
    global_addr_t   V_global_addr,
    global_addr_t   Kcache_global_addr,
    global_addr_t   Vcache_global_addr,
    global_addr_t   weight1_global_addr,
    global_addr_t   weight2_global_addr,
    global_addr_t   weight3_global_addr,
    global_addr_t   Y_global_addr,
    float           C,
    int             batch,
    int             hidden_size,
    int             num_attention_heads,
    int             num_k_v_heads,
    int             embeddings,
    int             attention_mode,
    data_type_t     dtype);

void tpu_kernel_llama_attention_multi_core(const void* api_buf) {

    sg_api_llama2_qkv_multi_core_t *api = (sg_api_llama2_qkv_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_llama2_qkv_multi_core(
        api->Q_global_addr,
        api->K_global_addr,
        api->V_global_addr,
        api->Kcache_global_addr,
        api->Vcache_global_addr,
        api->cos_global_addr,
        api->sin_global_addr,
        api->mask_global_addr,
        api->Y_global_addr,
        api->C,
        api->batch,
        api->hidden_size,
        api->num_attention_heads,
        api->num_k_v_heads,
        api->embeddings,
        api->attention_mode,
        (data_type_t)api->dtype);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_attention_multi_core);
#endif