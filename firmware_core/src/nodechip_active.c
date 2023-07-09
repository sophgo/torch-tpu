#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern void nodechip_active(
global_addr_t in_global_addr,
global_addr_t out_global_addr,
const int*    shape,
int           shape_dim,
data_type_t   dtype,
int           active_type,
float*        coef);

extern void nodechip_active_multi_core(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
const int*    shape,
int           dims,
int           active_type,
float*        coeff,
data_type_t   dtype);

void tpu_kernel_api_active ( const void * args )
{
  sg_api_active_t * api = ( sg_api_active_t * ) args;
  data_type_t dtype = ( data_type_t ) api->dtype;
  TPUKERNEL_ASSERT ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_active(
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    dtype,
    api->active_type,
    NULL);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_active );

void tpu_kernel_api_active_multi_core ( const void * args )
{
  sg_api_active_t * api = ( sg_api_active_t * ) args;
  data_type_t dtype = ( data_type_t ) api->dtype;
  TPUKERNEL_ASSERT ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_active_multi_core(
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    api->active_type,
    NULL,
    dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_active_multi_core );