#include "sg_api_struct.h"
#include "tpu_kernel.h"
// #include "config.h"

#ifdef BACKEND_SG2260

void firmware_kernel_tick();
void firmware_kernel_tock(int);

extern void nodechip_llama2_a16_matmul(
    global_addr_t activation_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t bias_global_addr,
    l2_sram_addr_t L2_addr,
    int final_col_num,
    int final_row_num,
    int inner_num,
    global_addr_t result_global_addr,
    global_addr_t zp_global_addr,
    global_addr_t scale_global_addr,
    int group_size,
    int weight_bits,
    data_type_t weight_dtype,
    data_type_t out_dtype,
    bool scale_zp_zip,
    bool has_bias
);

int tpu_kernel_api_llama_a16_matmul ( const void * args )
{
  sg_api_a16_matmul_t * api = ( sg_api_a16_matmul_t * ) args;
  TPUKERNEL_ASSERT((!api->has_bias || api->bias_dtype == api->io_dtype) && "for W x A mode, bias dtype has to be the same with io");
#ifdef USING_LLM_TICK_TOCK_PROFILE
  firmware_kernel_tick();
#endif

#ifndef REMOVE_POLLS_IN_LLM
  tpu_initialize();
#else
    tpu_poll_descriptor();
#endif

  nodechip_llama2_a16_matmul(
    api->input_global_addr,
    api->weight_global_addr,
    api->bias_global_addr,
    0,
    api->final_col_num,
    api->final_row_num,
    api->inner_num,
    api->output_global_addr,
    api->zp_global_addr,
    api->scale_global_addr,
    api->q_group_size,
    api->weight_bits,
    api->weight_dtype,
    api->io_dtype,
    true,
    api->has_bias
  );

#ifndef REMOVE_POLLS_IN_LLM
  tpu_poll();
#endif

#ifdef USING_LLM_TICK_TOCK_PROFILE
  firmware_kernel_tock(5);
#endif
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_llama_a16_matmul );

#endif
