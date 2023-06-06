#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"

extern void nodechip_concat_nd (
global_addr_t* input_global_addrs,
global_addr_t output_global_addr,
int ( *input_shapes ) [FW_MAX_SHAPE_DIMS],
int* st_by_concatway,
int dims,
int input_num,
int concat_axis,
data_type_t dtype );

void tpu_kernel_api_concat ( const void * args )
{
  sg_api_concat_t * api = ( sg_api_concat_t * ) args;
  int st_by_concatway[MAX_CONCATLAYER_INPUT_NUM] = { 0 };
  tpu_initialize();
  nodechip_concat_nd ( api->input_global_addrs,
                       api->output_global_addr,
                       api->input_shapes,
                       st_by_concatway,
                       api->shape_dim,
                       api->input_num,
                       api->concat_dim,
                       tpu_type_convert ( ( sg_data_type_t ) api->dtype ) );
  tpu_poll();
}
