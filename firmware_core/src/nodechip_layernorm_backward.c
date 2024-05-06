#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_layernorm_backward (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t mean_global_addr,
global_addr_t rstd_global_addr,
global_addr_t grad_input_global_addr,
global_addr_t grad_weight_global_addr,
global_addr_t grad_bias_global_addr,
int*          shape,
int           dims,
int           axis,
int           affine,
#ifdef BACKEND_SG2260
int           requires_grad_input,
#endif
data_type_t   dtype );

void tpu_kernel_api_layernorm_backward ( const void *args )
{
  sg_api_layernorm_backward_t *api = ( sg_api_layernorm_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_layernorm_backward(
    api->grad_output_global_addr,
    api->input_global_addr,
    api->weight_global_addr,
    api->mean_global_addr,
    api->rstd_global_addr,
    api->grad_input_global_addr,
    api->grad_weight_global_addr,
    api->grad_bias_global_addr,
    api->shape,
    api->dim,
    api->axis,
    api->grad_weight_global_addr && api->grad_bias_global_addr,
#ifdef BACKEND_SG2260
    api->requires_grad_input,
#endif
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_backward );

#ifdef BACKEND_SG2260
extern void nodechip_layernorm_backward_multi_core (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t mean_global_addr,
global_addr_t rstd_global_addr,
global_addr_t grad_input_global_addr,
global_addr_t grad_weight_global_addr,
global_addr_t grad_bias_global_addr,
int*          shape,
int           dims,
int           axis,
int           affine,
int            requires_grad_input,
data_type_t   dtype );

void tpu_kernel_api_layernorm_backward_multi_core ( const void *args )
{
  sg_api_layernorm_backward_t *api = ( sg_api_layernorm_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  nodechip_layernorm_backward_multi_core (
    api->grad_output_global_addr,
    api->input_global_addr,
    api->weight_global_addr,
    api->mean_global_addr,
    api->rstd_global_addr,
    api->grad_input_global_addr,
    api->grad_weight_global_addr,
    api->grad_bias_global_addr,
    api->shape,
    api->dim,
    api->axis,
    api->grad_weight_global_addr && api->grad_bias_global_addr,
    api->requires_grad_input,
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_backward_multi_core );
#endif