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
    data_type_t dtype);

void tpu_kernel_llama_mlp_multi_core(const void* api_buf) {

    sg_api_llama_mlp_multi_core_t *api = (sg_api_llama_mlp_multi_core_t*)api_buf;
    tpu_initialize();
    nodechip_llama_mlp_forward_multi_core(
        api->input_addr,
        api->weight0_addr,
        api->weight1_addr,
        api->weight2_addr,
        api->output_addr,
        api->batch,
        api->input_w,
        api->middle_w,
        (data_type_t)api->dtype);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_llama_mlp_multi_core);
#endif