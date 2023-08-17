#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern 
void nodechip_bcbinary_fp(
    global_addr_t A_global_addr,
    global_addr_t B_global_addr,
    global_addr_t res_global_addr,
    const int* A_shape,
    const int* B_shape,
    int A_dim,
    int B_dim,
    int binary_type,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

void tpu_kernel_api_div_eltwise ( const void *args )
{
  sg_api_div_eltwise_t *api = ( sg_api_div_eltwise_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_bcbinary_fp(
      api->input_global_addr,
      api->other_global_addr,
      api->output_global_addr,
      api->shape,
      api->shape,
      api->dim,
      api->dim,
      3, // 0:add, 1: sub, 2: mul ...
      (data_type_t)api->dtype,
      0,
      1);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_div_eltwise );
