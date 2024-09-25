#include "sg_api_struct.h"
#include "tpu_kernel.h"

void firmware_kernel_tick();
void firmware_kernel_tock(int);

#ifdef BACKEND_SG2260

extern
void nodechip_llama_attention_forward_multi_core(
    global_addr_t Q_global_addr,
    global_addr_t K_global_addr,
    global_addr_t V_global_addr,
    global_addr_t Qbuffer_global_addr,
    global_addr_t Kbuffer_global_addr,
    global_addr_t Vbuffer_global_addr,
    global_addr_t RoPE_cos_global_addr,
    global_addr_t RoPE_sin_global_addr,
    global_addr_t Mask_global_addr,
    global_addr_t Y_global_addr,
    global_addr_t Softmax_lse_global_addr,
    float C,
    float dropout_rate,
    int batch,
    int mask_max,
    int hidden_size,
    int num_attention_heads,
    int num_k_v_heads,
    int seq_len,
    data_type_t dtype,
    int qkv_packed,
    int return_softmax);

int tpu_kernel_llama_attention_forward_multi_core(const void* api_buf) {
    sg_api_llama_attention_forward_multi_core_t *api = (sg_api_llama_attention_forward_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_llama_attention_forward_multi_core(
        api-> Q_global_addr,
        api-> K_global_addr,
        api-> V_global_addr,
        api-> Qbuffer_global_addr,
        api-> Kbuffer_global_addr,
        api-> Vbuffer_global_addr,
        api-> cos_global_addr,
        api-> sin_global_addr,
        api-> mask_global_addr,
        api-> Y_global_addr,
        api-> Softmax_lse_global_addr,
        api-> C,
        api-> dropout_rate,
        api-> batch,
        api-> mask_max,
        api-> hidden_size,
        api-> num_attention_heads,
        api-> num_k_v_heads,
        api-> seq_len,
        (data_type_t)api-> dtype,
        api-> qkv_packed,
        api-> return_softmax);
    tpu_poll();
    return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_attention_forward_multi_core);

#endif
