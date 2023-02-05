#include "nodechip_conv_float_parallel.h"
#include "sg_api_struct.h"

void tpu_kernel_api_conv_forward(const void* args) {
    sg_api_conv_forward_t* api = (sg_api_conv_forward_t*)args;

    dim4 input_shape = {api->ishape[0], api->ishape[1],
                        api->ishape[2], api->ishape[3]};
    dim2 kernel = {api->kernel[0], api->kernel[1]};
    dim2 stride = {api->stride[0], api->stride[1]};
    dim2 dilation = {api->dilation[0], api->dilation[1]};
    padding_t pad = {api->pad[0], api->pad[1], api->pad[2], api->pad[3]};

    tpu_initialize();
    nodechip_conv_float_parallel(
        api->input_global_addr,
        api->weight_global_addr,
        api->bias_global_addr,
        api->output_global_addr,
        &input_shape,
        api->groups,
        api->output_c,
        &kernel,
        &stride,
        &dilation,
        &pad,
        api->has_bias == 1 ? true : false,
        api->if_relu == 1 ? true : false,
        api->upper_limit,
        api->result_add == 1 ? true : false,
        tpu_type_convert((sg_data_type_t)api->idtype),
        tpu_type_convert((sg_data_type_t)api->odtype),
        false);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conv_forward);
