#include "sg_api_struct.h"
#include "tpu_kernel.h"

void nodechip_arange (
int start,
int end,
int step,
global_addr_t output_global_addr,
int dtype,
const dim4  * output_shape
)
{
  local_addr_t output_addr;
  output_addr = 0;

  int todo = output_shape->c;
  dim4 output_local_shape_add = { .n = 1, .c = 65535, .h = 1, .w = 1};
  dim4 output_local_stride, output_global_stride;

  scalar_t C = { .s32 = 0 };
  while( todo > 0 )
  {
    output_local_shape_add.c = MIN( 65535, todo );
    tpu_compact_stride( &output_local_stride, 0, &output_local_shape_add);
    tpu_continuous_stride ( &output_global_stride, &output_local_shape_add );

    tpu_bdc_arithmetic_sequence_distribute(output_addr, start, step, output_local_shape_add.c );
    tpu_bdc_int_add_C( output_addr, output_addr, C, &output_local_shape_add, &output_local_stride, &output_local_stride, dtype, dtype, dtype, 0, RM_HALF_TO_EVEN, false );
    tpu_gdma_cpy_L2S ( output_global_addr, output_addr, &output_local_shape_add, &output_global_stride, &output_local_stride, dtype );
    output_global_addr += output_local_shape_add.c * tpu_data_type_size ( dtype );
    todo -= output_local_shape_add.c;
    C.s32 += output_local_shape_add.c * step;
  }
}

void tpu_kernel_api_arange ( const void * args )
{
  sg_api_arange_t * api = ( sg_api_arange_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_INT32);
  tpu_initialize();
  dim4 shape = { .n = 1, .c = api->shape[0], .h = 1, .w = 1 };
  nodechip_arange (
  api->start,
  api->end,
  api->step,
  api->output_global_addr,
  api->dtype,
  &shape);
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_arange );
