#include "sg_api_struct.h"
#include "tpu_kernel.h"


// softmax backward only support last dim for now
// if dtype is fp16/bf16, pooling will cast to fp32
extern void nodechip_softmax_backward_cast(
global_addr_t grad_input_global_addr,
global_addr_t grad_output_global_addr,
global_addr_t output_global_addr,
const int*    shape,
int           dims,
int           axis,
data_type_t   dtype );

void tpu_kernel_api_softmax_backward ( const void *args )
{
  sg_api_softmax_backward_t *api = ( sg_api_softmax_backward_t * ) args;
  tpu_initialize();
  TPUKERNEL_ASSERT ( api->axis == api->dim - 1 );
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  nodechip_softmax_backward_cast (
  api->grad_input_global_addr,
  api->grad_output_global_addr,
  api->output_global_addr,
  api->shape,
  api->dim,
  api->axis,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax_backward );

#ifdef BACKEND_SG2260
extern void nodechip_softmax_backward_multi_core (
  global_addr_t grad_input_global_addr,
  global_addr_t grad_output_global_addr,
  global_addr_t output_global_addr,
  int*          shape,
  int           axis,
  float         scale_val,
  data_type_t   dtype );

void tpu_kernel_api_softmax_backward_multi_core ( const void *args )
{
  sg_api_softmax_backward_t *api = ( sg_api_softmax_backward_t * ) args;
  tpu_initialize();
  TPUKERNEL_ASSERT ( api->axis == api->dim - 1 );
  nodechip_softmax_backward_multi_core (
    api->grad_input_global_addr,
    api->grad_output_global_addr,
    api->output_global_addr,
    api->shape,
    api->axis,
    1.f,
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax_backward_multi_core );
#endif