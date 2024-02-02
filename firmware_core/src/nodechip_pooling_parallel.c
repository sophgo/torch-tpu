#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_pooling_parallel_with_data_split(
    global_addr_t      ifmap_offset_global,
    global_addr_t      ofmap_offset_global,
    int             input_n,
    int             input_c,
    int             input_h,
    int             input_w,
    int             output_h,
    int             output_w,
    int             kh,
    int             kw,
    int             pad_h,
    int             pad_w,
    int             pad_h_after,
    int             pad_w_after,
    int             stride_h,
    int             stride_w,
    int             dilation_h,
    int             dilation_w,
    int             is_avg_pooling,
    int             avg_pooling_mode,
    int             if_relu,
    float           relu_upper_limit,
    data_type_t     dtype
);

void tpu_kernel_api_avg_pooling(const void *args)
{
  sg_api_pooling_t *api = (sg_api_pooling_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);

  tpu_initialize();
  nodechip_pooling_parallel_with_data_split(
        api->input_global_addr,
        api->output_global_addr,
        api->input_shape[0],
        api->input_shape[1],
        api->input_shape[2],
        api->input_shape[3],
        api->output_shape[2],
        api->output_shape[3],
        api->pooling_desc.kh,
        api->pooling_desc.kw,
        api->pooling_desc.pad_h,
        api->pooling_desc.pad_w,
        api->pooling_desc.pad_h,
        api->pooling_desc.pad_w,
        api->pooling_desc.stride_h,
        api->pooling_desc.stride_w,
        1/*dilation_h*/, 1/*dilation_w*/,
        api->pooling_desc.mode == POOLING_AVG,
        0/*avg_pooling_mode*/, 0/*if_relu*/,
        -1/*relu_upper_limit*/, api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_avg_pooling);

#ifdef BACKEND_SG2260
void tpu_kernel_api_avg_pooling_multi_core(const void *args)
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_avg_pooling_multi_core);
#endif