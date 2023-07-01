#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern void nodechip_cast (
global_addr_t   in_global_addr,
global_addr_t   out_global_addr,
const int*      shape,
int             shape_dim,
data_type_t     src_dtype,
data_type_t     dst_dtype,
rounding_mode_t round_mode );

void tpu_kernel_api_dtype_convert ( const void* args ) {
  sg_api_dtype_convert_t* api = ( sg_api_dtype_convert_t* ) args;
  tpu_initialize();
  nodechip_cast (
  api->input_global_addr,
  api->output_global_addr,
  api->shape,
  api->dim,
  ( data_type_t ) api->input_dtype,
  ( data_type_t ) api->output_dtype,
  RM_HALF_TO_EVEN );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_dtype_convert );
