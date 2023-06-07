#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"

extern void nodechip_index_select (
global_addr_t input_global_addr,
global_addr_t index_global_addr,
global_addr_t output_global_addr,
const int     *input_shape,
int           shape_dims,
int           index_num,
int           axis, // axis to do index_select
int           const_val, // fill_value if index not found in input
data_type_t   dtype );

void tpu_kernel_api_index_select ( const void * args )
{
  sg_api_index_select_t *api = ( sg_api_index_select_t * ) args;
  tpu_initialize();
  int input_shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  for ( int i = 0; i < api->shape_dims; ++i ) {
    input_shape[i] = api->input_shape[i];
  }
  nodechip_index_select (
  api->input_global_addr,
  api->index_global_addr,
  api->output_global_addr,
  input_shape,
  api->shape_dims,
  api->index_num,
  api->axis,
  api->const_val,
  tpu_type_convert ( api->dtype ) );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_index_select );
