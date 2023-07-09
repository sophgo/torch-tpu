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
data_type_t   dtype );

void nodechip_layernorm_backward_multi_core (
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
data_type_t   dtype )
{
  int slice_num = tpu_core_num();
  int slice_idx = tpu_core_index();
  TPUKERNEL_ASSERT ( slice_num > 0 );
  TPUKERNEL_ASSERT ( 0 <= slice_idx && slice_idx < slice_num );
  int outer_num = 1, inner_num = 1;
  for (int i = 0; i < axis; i++)
  {
      outer_num *= shape[i];
  }
  for (int i = axis; i < dims; i++)
  {
      inner_num *= shape[i];
  }
  int slice = DIV_UP (outer_num, slice_num);
  int offset = slice_idx * slice * inner_num;
  int param_offset = slice_idx * slice;
  if( slice * (slice_idx + 1) > outer_num )
  {
    slice = outer_num - slice * slice_idx;
  }

#if defined(__sg2260__) && USING_L2
  TPUKERNEL_ASSERT ( 2 * inner_num * tpu_data_type_size(dtype) < L2_SRAM_SIZE );
  scalar_t Zero = { .u32 = 0 };
  tpu_sdma_system_set ( L2_SRAM_START_ADDR, Zero, 2 * inner_num, DT_FP32 );
  tpu_sync_all();
#endif

  if ( slice > 0 )
  {
    int dsize = tpu_data_type_size(dtype);
    int multi_core_shape[] = {slice, inner_num};
    nodechip_layernorm_backward(
      grad_output_global_addr + offset * dsize,
      input_global_addr + offset * dsize,
      weight_global_addr,
      mean_global_addr + param_offset * dsize,
      rstd_global_addr + param_offset * dsize,
      grad_input_global_addr + offset * dsize,
      grad_weight_global_addr,
      grad_bias_global_addr,
      multi_core_shape,
      2,
      1,
      1,
      dtype);
  }

#if defined(__sg2260__) && USING_L2
  tpu_sync_all();
  if ( slice > 0 )
  {
    tpu_sdma_system_cpy (grad_bias_global_addr, L2_SRAM_START_ADDR, inner_num, dtype);
    tpu_sdma_system_cpy (grad_weight_global_addr, L2_SRAM_START_ADDR + inner_num * tpu_data_type_size(dtype), inner_num, dtype);
  }
#endif

  tpu_sync_all();
}

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
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_backward );

void tpu_kernel_api_layernorm_backward_multi_core ( const void *args )
{
  sg_api_layernorm_backward_t *api = ( sg_api_layernorm_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
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
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_backward_multi_core );