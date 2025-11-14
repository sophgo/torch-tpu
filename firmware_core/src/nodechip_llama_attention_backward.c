#include "sg_api_struct.h"
#include "tpu_kernel.h"

#ifdef ENABLE_MULTI_CORE
// #ifdef 1
extern
void nodechip_llama2_qkv_backward_multi_core(
    global_addr_t Q_global_addr,
    global_addr_t K_global_addr,
    global_addr_t V_global_addr,
    global_addr_t O_global_addr,
    global_addr_t L_global_addr,
    global_addr_t dO_global_addr,
    global_addr_t dQ_global_addr,
    global_addr_t dK_global_addr,
    global_addr_t dV_global_addr,
    global_addr_t RoPE_cos_global_addr,
    global_addr_t RoPE_sin_global_addr,
    global_addr_t mask_global_addr,
    global_addr_t input_length_global_addr,
    float C,
    int batch,
    int mask_max,
    int hidden_size,
    int num_attention_heads,
    int num_k_v_heads,
    data_type_t dtype);

int tpu_kernel_llama_attention_backward_multi_core(const void* api_buf) {

    sg_api_llama2_qkv_backward_multi_core_t *api = (sg_api_llama2_qkv_backward_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_llama2_qkv_backward_multi_core(
        api-> Q_global_addr,
        api-> K_global_addr,
        api-> V_global_addr,
        api-> O_global_addr,
        api-> l_global_addr,
        api-> dO_global_addr,
        api-> dQ_global_addr,
        api-> dK_global_addr,
        api-> dV_global_addr,
        api-> cos_global_addr,
        api-> sin_global_addr,
        api-> mask_global_addr,
        api-> input_lengths_global_addr,
        api-> C,
        api-> batch,
        api-> mask_max,
        api-> hidden_size,
        api-> q_heads,
        api-> kv_heads,
        (data_type_t)api-> dtype);
    tpu_poll();
    return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_attention_backward_multi_core);
#endif