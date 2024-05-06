#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_clip_fp(
    global_addr_t A_global_addr,
    global_addr_t res_global_addr,
    const int* shape,
    int shape_dim,
    float min,
    float max,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

void tpu_kernel_api_clamp ( const void *args )
{
  sg_api_clamp_t *api = ( sg_api_clamp_t * ) args;
  tpu_initialize();
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int shape[api->dim];
  for (int i = 0; i < api->dim; ++i)
  {
    shape[i] = api->shape[i];
  }
  nodechip_clip_fp (
  api->input_global_addr,
  api->output_global_addr,
  shape,
  api->dim,
  api->min,
  api->max,
  ( data_type_t )api->dtype,
  /*if_relu*/0,
  /*relu_upper_limit*/0.0f );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_clamp );

#ifdef BACKEND_SG2260
void tpu_kernel_api_clamp_multi_core ( const void *args )
{
  sg_api_clamp_t * api = ( sg_api_clamp_t * ) args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  int length = 1;
  int shape[api->dim];
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
    shape[i] = api->shape[i];
  }
  tpu_initialize();

  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  nodechip_clip_fp(
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
      shape,
      api->dim,
      api->min,
      api->max,
      ( data_type_t )api->dtype,
      /*if_relu*/0,
      /*relu_upper_limit*/0.0f );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_clamp_multi_core );
#endif