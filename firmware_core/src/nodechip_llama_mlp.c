#include "sg_api_struct.h"
#include "tpu_kernel.h"


#ifdef BACKEND_SG2260
extern
void nodechip_llama_mlp_forward_multi_core(
    /*
        input: [batch, input_w]
        weight0: [input_w, middle_w]
        weight1: [input_w, middle_w]
        weight2: [middle_w, input_w]
        ouput: [batch, input_w]

        formula: matmul(mul(matmul(input, weight0), mul(matmul(input, weight1), sigmoid(matmul(input, weight1))), weight2)
    */
    global_addr_t input_global_addr,
    global_addr_t weight0_global_addr,
    global_addr_t weight1_global_addr,
    global_addr_t weight2_global_addr,
    global_addr_t output_global_addr,
    int batch,
    int input_w,
    int middle_w,
    data_type_t dtype,
    int enable_8ch,
    int input_slice_dim,
    int weight0_slice_dim,
    int weight1_slice_dim,
    int weight2_slice_dim,
    int output_slice_dim,
    global_addr_t input_8ch_global_addr[8],
    global_addr_t weight0_8ch_global_addr[8],
    global_addr_t weight1_8ch_global_addr[8],
    global_addr_t weight2_8ch_global_addr[8],
    global_addr_t output_8ch_global_addr[8],
    bool quantized,//dequant param
    int group_size,
    int weight_bits,
    global_addr_t zp0_global_addr,
    global_addr_t scale0_global_addr,
    global_addr_t zp1_global_addr,
    global_addr_t scale1_global_addr,
    global_addr_t zp2_global_addr,
    global_addr_t scale2_global_addr);

    // void set_file_dump_subdir(const char*);
void tpu_kernel_llama_mlp_multi_core(const void* api_buf) {
    // set_file_dump_subdir("./Custom_MLP");
    sg_api_llama_mlp_multi_core_t *api = (sg_api_llama_mlp_multi_core_t*)api_buf;
    tpu_initialize();
#ifdef USING_PERF_MODE
        tpu_sync_all();
#endif
    global_addr_t *rmsnorm_useless = NULL;
    nodechip_llama_mlp_forward_multi_core(
        api->input_addr,
        api->weight0_addr,
        api->weight1_addr,
        api->weight2_addr,
        api->output_addr,
        api->batch,
        api->input_w,
        api->middle_w,
        (data_type_t)api->dtype,
        0,
        -1,
        -1,
        -1,
        -1,
        -1,
        rmsnorm_useless,
        rmsnorm_useless,
        rmsnorm_useless,
        rmsnorm_useless,
        rmsnorm_useless,
        api->quantized,
        api->group_size,
        api->weight_bits,
        api->zp0_addr,
        api->scale0_addr,
        api->zp1_addr,
        api->scale1_addr,
        api->zp2_addr,
        api->scale2_addr);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_mlp_multi_core);
#endif