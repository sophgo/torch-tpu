#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_batch_matmul_float (
  global_addr_t L_global_addr,
  global_addr_t R_global_addr,
  global_addr_t bias_global_addr,
  global_addr_t rescale_global_addr,
  global_addr_t Y_global_addr,
  data_type_t left_dtype,
  data_type_t right_dtype,
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
  float relu_upper_limit,
  bool do_rescale,
  bool rescale_is_const,
  int rescale_const_val);

int tpu_kernel_api_batch_matmul ( const void * args )
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
                                0xffffffff,
                                api->output_global_addr,
                                ( data_type_t ) api->dtype,
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
                                0,
                                false,
                                false,
                                0);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_batch_matmul );

#ifdef BACKEND_SG2260
extern void nodechip_matmul_multi_core(
    global_addr_t   left_global_addr,
    global_addr_t   right_global_addr,
    global_addr_t   bias_global_addr,
    global_addr_t   output_global_addr,
    const int*      L_shape,
    const int*      R_shape,
    int             L_dims,
    int             R_dims,
    int             L_trans,
    int             R_trans,
    data_type_t     in_dtype,
    data_type_t     out_dtype,
    int             slice_core_m,
    int             slice_core_n,
    int             slice_m,
    int             slice_n,
    int             slice_k,
    int             slyt_num,
    int             left_slyt_fmt,
    int             right_slyt_fmt,
    int             result_slyt_fmt,
    global_addr_t*  ga_private_left,
    global_addr_t*  ga_private_right,
    global_addr_t*  ga_private_result);

int tpu_kernel_api_matmul_multi_core(const void* api_buf) {
    sg_api_matmul_multi_core_t *api = (sg_api_matmul_multi_core_t *)api_buf;
    tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
    nodechip_matmul_multi_core(
        api->left_global_addr,
        api->right_global_addr,
        api->bias_global_addr,
        api->output_global_addr,
        api->L_shape,
        api->R_shape,
        api->L_dims,
        api->R_dims,
        api->L_trans,
        api->R_trans,
        (data_type_t)api->in_dtype,
        (data_type_t)api->out_dtype,
        api->slice_core_m,
        api->slice_core_n,
        api->slice_m,
        api->slice_n,
        api->slice_k,
        api->slyt_num,
        api->left_slyt_fmt,
        api->right_slyt_fmt,
        api->result_slyt_fmt,
        api->left_slyt_global_addr,
        api->right_slyt_global_addr,
        api->result_slyt_global_addr);
    tpu_poll();
    return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_matmul_multi_core);
#endif