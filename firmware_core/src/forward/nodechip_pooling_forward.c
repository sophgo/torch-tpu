#include "nodechip_pooling_parallel.h"
#include "sg_api_struct.h"

void tpu_kernel_api_pooling_forward(const void* args) {
    sg_api_pooling_forward_t* api = (sg_api_pooling_forward_t*)args;

    tpu_initialize();
    if (api->max_with_mask == 1) {
        nodechip_max_pooling_with_mask_forward(
            api->input_global_addr,
            api->output_global_addr,
            api->max_mask_global_addr,
            api->input_n,
            api->input_c,
            api->input_h,
            api->input_w,
            api->output_h,
            api->output_w,
            api->kh,
            api->kw,
            api->pad_h,
            api->pad_w,
            api->pad_h_after,
            api->pad_w_after,
            api->stride_h,
            api->stride_w,
            api->dilation_h,
            api->dilation_w,
            api->is_avg_pooling,
            api->avg_pooling_mode,
            api->if_relu,
            api->relu_upper_limit,
            tpu_type_convert(api->data_type));
    } else {
        nodechip_pooling_parallel_with_data_split(
            api->input_global_addr,
            api->output_global_addr,
            api->input_n,
            api->input_c,
            api->input_h,
            api->input_w,
            api->output_h,
            api->output_w,
            api->kh,
            api->kw,
            api->pad_h,
            api->pad_w,
            api->pad_h_after,
            api->pad_w_after,
            api->stride_h,
            api->stride_w,
            api->dilation_h,
            api->dilation_w,
            api->is_avg_pooling,
            api->avg_pooling_mode,
            api->if_relu,
            api->relu_upper_limit,
            tpu_type_convert(api->data_type));
    }
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pooling_forward);
