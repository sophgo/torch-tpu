#include "nodechip_contiguous.h"

void nodechip_strided_copy(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    int           dim,
    const int*    shape,
    const int*    in_stride,
    const int*    out_stride,
    data_type_t   dtype)
{
    dim4 copy_shape = {.n=1,.c =1, .h =1, .w =1};
    dim4 copy_in_stride = {.n=1,.c =1, .h =1, .w =1};
    dim4 copy_out_stride = {.n=1,.c =1, .h =1, .w =1};
    TPUKERNEL_ASSERT(dim >=1 && dim <=4);

    if (dim >= 1){
        copy_shape.n = shape[0];
        copy_in_stride.n = in_stride[0];
        copy_out_stride.n = out_stride[0];
    }
    if (dim >= 2){
        copy_shape.c = shape[1];
        copy_in_stride.c = in_stride[1];
        copy_out_stride.c = out_stride[1];
    }
    if (dim >= 3){
        copy_shape.h = shape[2];
        copy_in_stride.h = in_stride[2];
        copy_out_stride.h = out_stride[2];
    }
    if (dim >= 4){
        copy_shape.w = shape[3];
        copy_in_stride.w = in_stride[3];
        copy_out_stride.w = out_stride[3];
    }
    TPUKERNEL_ASSERT(copy_in_stride.w <= 128 / tpu_data_type_size(dtype) &&
                        copy_out_stride.w <= 128 / tpu_data_type_size(dtype));
    tpu_gdma_cpy_S2S(
        out_global_addr,
        in_global_addr,
        &copy_shape,
        &copy_out_stride,
        &copy_in_stride,
        dtype);

}

void nodechip_contiguous(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    int           dim,
    const int*    shape,
    const int*    in_stride,
    data_type_t   dtype)
{
    int out_stride[FW_MAX_SHAPE_DIMS] = {0};
    int stride = 1;
    for (int i = dim-1; i >= 0; i--){
        out_stride[i] = stride;
        stride *= shape[i];
    }
    nodechip_strided_copy(
        in_global_addr,
        out_global_addr,
        dim,
        shape,
        in_stride,
        out_stride,
        dtype);
}

void tpu_kernel_api_contiguous_forward(const void *args)
{
    sg_api_contiguous_forward_t *api = (sg_api_contiguous_forward_t*)args;
    tpu_initialize();
    nodechip_contiguous(
        api->in_global_addr,
        api->out_global_addr,
        api->shape_dim,
        api->shape,
        api->stride,
        tpu_type_convert((sg_data_type_t)api->dtype)
        );
    tpu_poll();
}

void tpu_kernel_api_strided_copy(const void *args)
{
    sg_api_strided_copy_t *api = (sg_api_strided_copy_t*)args;
    tpu_initialize();
    nodechip_strided_copy(
        api->in_global_addr,
        api->out_global_addr,
        api->shape_dim,
        api->shape,
        api->in_stride,
        api->out_stride,
        tpu_type_convert((sg_data_type_t)api->dtype)
        );
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_contiguous_forward);
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_strided_copy);