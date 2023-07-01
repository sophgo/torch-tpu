#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern void nodechip_batch_matmul_float (
global_addr_t L_global_addr,
global_addr_t R_global_addr,
global_addr_t bias_global_addr,
global_addr_t Y_global_addr,
data_type_t in_dtype,
data_type_t out_dtype,
const int* L_shape,
const int* R_shape,
int L_dim,
int R_dim,
int* Y_shape,
int* Y_dim,
int L_trans,
int R_trans,
int hdim_is_batch,
int has_bias,
bool do_relu,
float relu_upper_limit );

void tpu_kernel_api_batch_matmul ( const void * args )
{
  sg_api_batch_matmul_t * api = ( sg_api_batch_matmul_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  int left_shape[] =
  {
    api->batch,
    api->is_left_transposed ? api->left_column : api->left_row,
    api->is_left_transposed ? api->left_row : api->left_column
  };
  int right_shape[] =
  {
    api->batch,
    api->is_right_transposed ? api->right_column : api->left_column,
    api->is_right_transposed ? api->left_column : api->right_column
  };
  nodechip_batch_matmul_float ( api->left_global_addr,
                                api->right_global_addr,
                                api->bias_global_addr,
                                api->output_global_addr,
                                ( data_type_t ) api->dtype,
                                ( data_type_t ) api->dtype,
                                left_shape,
                                right_shape,
                                3,
                                3,
                                0,
                                0,
                                api->is_left_transposed,
                                api->is_right_transposed,
                                0,
                                api->bias_global_addr != 0,
                                0,
                                0 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_batch_matmul );
