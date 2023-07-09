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

void nodechip_gelu_forward_multi_core (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int*          shape,
int           dims,
data_type_t   dtype )
{
  int slice_num = tpu_core_num();
  int slice_idx = tpu_core_index();
  TPUKERNEL_ASSERT ( slice_num > 0 );
  TPUKERNEL_ASSERT ( 0 <= slice_idx && slice_idx < slice_num );
  int numel = 1;
  for ( int i = 0; i < dims; i++ )
  {
    numel *= shape[i];
  }
  int slice = DIV_UP ( numel, slice_num );
  if( slice * (slice_idx + 1) > numel )
  {
    slice = numel - slice * slice_idx;
  }
  int offset = slice_idx * DIV_UP ( numel, slice_num );
  if ( slice > 0 )
  {
    const int dsize = tpu_data_type_size( dtype );
    nodechip_active(
      input_global_addr + offset * dsize,
      output_global_addr + offset * dsize,
      &slice,
      1,
      dtype,
      29,//ACTIVE_GELU
      NULL);
  }
  tpu_sync_all();
}

void tpu_kernel_api_gelu ( const void * args )
{
  sg_api_gelu_t * api = ( sg_api_gelu_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_active(
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    ( data_type_t ) api->dtype,
    29,//ACTIVE_GELU
    NULL);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gelu );

void tpu_kernel_api_gelu_multi_core ( const void * args )
{
  sg_api_gelu_t * api = ( sg_api_gelu_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_gelu_forward_multi_core (
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gelu_multi_core );