#include "sg_api_struct.h"
#include "tpu_kernel.h"


// if dtype is fp16/bf16, pooling will cast to fp32
extern void nodechip_layernorm_forward_cast (
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t bias_global_addr,
global_addr_t output_global_addr,
global_addr_t mean_global_addr,
global_addr_t rstd_global_addr,
const int*    shape,
int           dims,
int           axis,
float         eps,
bool          affine,
data_type_t   dtype );

int tpu_kernel_api_layernorm ( const void *args )
{
  sg_api_layernorm_t *api = ( sg_api_layernorm_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_layernorm_forward_cast (
    api->input_global_addr,
    api->weight_global_addr,
    api->bias_global_addr,
    api->output_global_addr,
    api->mean_global_addr,
    api->rstd_global_addr,
    api->shape,
    api->dim,
    api->axis,
    api->eps,
    api->weight_global_addr && api->bias_global_addr,
    ( data_type_t ) api->dtype );
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm );

#ifdef BACKEND_SG2260
extern void nodechip_layernorm_forward_multi_core (
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t bias_global_addr,
global_addr_t mean_global_addr,
global_addr_t rstd_global_addr,
global_addr_t output_global_addr,
int*          shape,
int           dims,
int           axis,
float         eps,
int           affine,
data_type_t   dtype );
int tpu_kernel_api_layernorm_multi_core ( const void *args )
{
  sg_api_layernorm_t *api = ( sg_api_layernorm_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
#ifdef USING_PERF_MODE
  tpu_sync_all();
#endif
  nodechip_layernorm_forward_multi_core (
    api->input_global_addr,
    api->weight_global_addr,
    api->bias_global_addr,
    api->mean_global_addr,
    api->rstd_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    api->axis,
    api->eps,
    api->weight_global_addr && api->bias_global_addr,
    ( data_type_t ) api->dtype );
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_multi_core );
#endif