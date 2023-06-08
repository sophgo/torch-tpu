#include "nodechip_softmax.h"
#include "sg_api_struct.h"

extern void nodechip_softmax_forward_2DR1_parallel (
global_addr_t IGAddr,
global_addr_t OGAddr,
int Row, // row number
int Column, // column number
data_type_t DType ); // DT_FP32 or DT_FP16

void tpu_kernel_api_softmax_forward ( const void *args )
{
  sg_api_softmax_forward_t *api = ( sg_api_softmax_forward_t * ) args;
  tpu_initialize();
  if ( api->compute_dim == api->dims - 1 && api->scale_val == 1.f )
  {
    int Row = 1;
    for ( int i = 0; i < api->dims - 1; ++i )
    {
      Row *= api->shape[i];
    }
    int Column = api->shape[api->dims - 1];
    nodechip_softmax_forward_2DR1_parallel (
    api->input_global_addr,
    api->output_global_addr,
    Row,
    Column,
    tpu_type_convert ( api->dtype ) );
  }
  else
  {
    nodechip_softmax (
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dims,
    api->compute_dim,
    api->compute_dim,
    0,
    api->scale_val,
    tpu_type_convert ( api->dtype ) );
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax_forward );
