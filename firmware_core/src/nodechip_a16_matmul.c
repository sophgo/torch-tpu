#include "sg_api_struct.h"
#include "tpu_kernel.h"
// #include "config.h"

#ifdef BACKEND_SG2260

extern void nodechip_llama2_a16_matmul(
    global_addr_t activation_global_addr,
    global_addr_t weight_global_addr,
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
    bool scale_zp_zip
);

int tpu_kernel_api_llama_a16_matmul ( const void * args )
{
  sg_api_a16_matmul_t * api = ( sg_api_a16_matmul_t * ) args;
  TPUKERNEL_ASSERT((!api->has_bias || api->bias_dtype == api->io_dtype) && "for W x A mode, bias dtype has to be the same with io");
  tpu_initialize();
  nodechip_llama2_a16_matmul(
    api->input_global_addr,
    api->weight_global_addr,
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
    true
  );
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_llama_a16_matmul );

#endif
