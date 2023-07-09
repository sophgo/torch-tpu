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

void nodechip_layernorm_forward_multi_core (
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t bias_global_addr,
global_addr_t output_global_addr,
global_addr_t mean_global_addr,
global_addr_t rstd_global_addr,
int*          shape,
int           dims,
int           axis,
float         eps,
data_type_t   dtype )
{
  int slice_num = tpu_core_num();
  int slice_idx = tpu_core_index();
  TPUKERNEL_ASSERT ( slice_num > 0 );
  TPUKERNEL_ASSERT ( 0 <= slice_idx && slice_idx < slice_num );
  int inner_num = 1, outer_num = 1;
  for (int i = 0; i < dims; ++i)
  {
      if(i < axis) { outer_num *= shape[i]; }
      if(i >= axis) { inner_num *= shape[i]; }
  }
  int slice = DIV_UP (outer_num, slice_num);
  int offset = slice_idx * slice * inner_num;
  int param_offset = slice_idx * slice;
  if( slice * (slice_idx + 1) > outer_num )
  {
    slice = outer_num - slice * slice_idx;
  }
  if ( slice > 0 )
  {
    const int dsize = tpu_data_type_size(dtype);
    const int multi_core_shape[] = {slice, inner_num};
    nodechip_layernorm_forward_cast(
      input_global_addr + offset * dsize,
      weight_global_addr,
      bias_global_addr,
      output_global_addr + offset * dsize,
      mean_global_addr + param_offset * dsize,
      rstd_global_addr + param_offset * dsize,
      multi_core_shape,
      2,
      1,
      eps,
      1,
      dtype);
  }
  tpu_sync_all();
}

void tpu_kernel_api_layernorm ( const void *args )
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
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm );

void tpu_kernel_api_layernorm_multi_core ( const void *args )
{
  sg_api_layernorm_t *api = ( sg_api_layernorm_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_layernorm_forward_multi_core (
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
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_multi_core );
