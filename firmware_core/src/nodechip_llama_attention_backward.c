#include "sg_api_struct.h"
#include "tpu_kernel.h"

#ifdef BACKEND_SG2260
// #ifdef 1
extern
void nodechip_llama2_qkv_backward_multi_core(
    global_addr_t Q_global_addr,
    global_addr_t K_global_addr,
    global_addr_t V_global_addr,
    global_addr_t O_global_addr,
    global_addr_t L_global_addr,
    global_addr_t dO_global_addr,
    global_addr_t Qbuffer_global_addr,
    global_addr_t Kbuffer_global_addr,
    global_addr_t Vbuffer_global_addr,
    global_addr_t Obuffer_global_addr,
    global_addr_t Lbuffer_global_addr,
    global_addr_t dObuffer_global_addr,
    global_addr_t dQ_global_addr,
    global_addr_t dK_global_addr,
    global_addr_t dV_global_addr,
    global_addr_t RoPE_cos_global_addr,
    global_addr_t RoPE_sin_global_addr,
    global_addr_t input_length_global_addr,
    float C,
    int batch,
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
        api-> Qbuffer_global_addr,
        api-> Kbuffer_global_addr,
        api-> Vbuffer_global_addr,
        api-> Obuffer_global_addr,
        api-> lbuffer_global_addr,
        api-> dObuffer_global_addr,
        api-> dQ_global_addr,
        api-> dK_global_addr,
        api-> dV_global_addr,
        api-> cos_global_addr,
        api-> sin_global_addr,
        api-> input_lengths_global_addr,
        api-> C,
        api-> batch,
        api-> hidden_size,
        api-> q_heads,
        api-> kv_heads,
        (data_type_t)api-> dtype);
    tpu_poll();
    return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_attention_backward_multi_core);
#endif