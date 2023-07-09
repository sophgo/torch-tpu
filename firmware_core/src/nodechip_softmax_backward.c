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

void nodechip_softmax_backward_multi_core (
global_addr_t grad_input_global_addr,
global_addr_t grad_output_global_addr,
global_addr_t output_global_addr,
int*          shape,
int           dims,
int           axis,
data_type_t   dtype )
{
  int slice_num = tpu_core_num();
  int slice_idx = tpu_core_index();
  TPUKERNEL_ASSERT ( slice_num > 0 );
  TPUKERNEL_ASSERT ( 0 <= slice_idx && slice_idx < slice_num );
  int outer_num = 1;
  for (int i = 0; i < axis; i++)
  {
      outer_num *= shape[i];
  }
  int inner_num = shape[axis];
  int slice = DIV_UP (outer_num, slice_num);
  int offset = slice_idx * slice * inner_num;
  if( slice * (slice_idx + 1) > outer_num )
  {
    slice = outer_num - slice * slice_idx;
  }
  if ( slice > 0 )
  {
    int dsize = tpu_data_type_size( dtype );
    int multi_core_shape[] = {slice, inner_num};
    nodechip_softmax_backward_cast(
      grad_input_global_addr + offset * dsize,
      grad_output_global_addr + offset * dsize,
      output_global_addr + offset * dsize,
      multi_core_shape,
      2,
      1,
      dtype);
  }
  tpu_sync_all();
}

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
    api->dim,
    api->axis,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax_backward_multi_core );
