#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern
void nodechip_llama2_attention(
    global_addr_t Q_global_addr,
    global_addr_t K_global_addr,
    global_addr_t V_global_addr,
    global_addr_t Kcache_global_addr,
    global_addr_t Vcache_global_addr,
    global_addr_t Mask_global_addr,
    global_addr_t Y_global_addr,
    global_addr_t input_length_global_addr,
    global_addr_t save_slots_global_addr,
    global_addr_t fetch_slots_global_addr,
    int slots_size,
    float C,
    int batch,
    int mask_max,
    int head_size,
    int num_attention_heads,
    int num_k_v_heads,
    int attention_mode,
    int block_size,
    data_type_t dtype,
    int qkv_packed);

int tpu_kernel_llama_attention(const void* api_buf) {
    sg_api_llama2_qkv_t *api = (sg_api_llama2_qkv_t*)api_buf;
    tpu_initialize();
    nodechip_llama2_attention(
        api->Q_global_addr,
        api->K_global_addr,
        api->V_global_addr,
        api->Kcache_global_addr,
        api->Vcache_global_addr,
        api->mask_global_addr,
        api->OUT_global_addr,
        api->input_lengths_global_addr,
        api->save_slots_global_addr,
        api->fetch_slots_global_addr,
        api->slots_size,
        api->C,
        api->batch,
        api->mask_size,
        api->head_size,
        api->q_heads,
        api->kv_heads,
        api->attention_mode,
        api->block_size,
        (data_type_t)api->dtype,
	api->qkv_packed);
    tpu_poll();
    return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_attention);

#ifdef BACKEND_SG2260
// #ifdef 1
extern

void nodechip_llama2_qkv_multi_core(
    global_addr_t Q_global_addr,
    global_addr_t K_global_addr,
    global_addr_t V_global_addr,
    global_addr_t Qbuffer_global_addr,
    global_addr_t Kbuffer_global_addr,
    global_addr_t Vbuffer_global_addr,
    global_addr_t Kcache_global_addr,
    global_addr_t Vcache_global_addr,
    global_addr_t RoPE_cos_global_addr,
    global_addr_t RoPE_sin_global_addr,
    global_addr_t Mask_global_addr,
    global_addr_t Y_global_addr,
    global_addr_t input_length_global_addr,
    global_addr_t save_slots_global_addr,
    global_addr_t fetch_slots_global_addr,
    int slots_size,
    float C,
    int batch,
    int mask_max,
    int hidden_size,
    int num_attention_heads,
    int num_k_v_heads,
    int embeddings,
    int attention_mode,
    int block_size,
    data_type_t dtype,
    int qkv_packed);

int tpu_kernel_llama_attention_multi_core(const void* api_buf) {

    sg_api_llama2_qkv_multi_core_t *api = (sg_api_llama2_qkv_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_llama2_qkv_multi_core(
        api-> Q_global_addr,
        api-> K_global_addr,
        api-> V_global_addr,
        api-> Qbuffer_global_addr,
        api-> Kbuffer_global_addr,
        api-> Vbuffer_global_addr,
        api-> Kcache_global_addr,
        api-> Vcache_global_addr,
        api-> cos_global_addr,
        api-> sin_global_addr,
        api-> mask_global_addr,
        api-> OUT_global_addr,
        api-> input_lengths_global_addr,
        api-> save_slots_global_addr,
        api-> fetch_slots_global_addr,
        api-> slots_size,
        api-> C,
        api-> batch,
        api-> mask_size,
        api-> hidden_size,
        api-> q_heads,
        api-> kv_heads,
        api-> hidden_size,
        api-> attention_mode,
        api-> block_size,
        (data_type_t)api-> dtype,
        api-> qkv_packed);
    tpu_poll();
    return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_attention_multi_core);
#endif