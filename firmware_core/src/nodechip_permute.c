#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_transpose(global_addr_t         input_global_addr,
                               global_addr_t         output_global_addr,
                               int                   *input_shape,
                               int                   *order,
                               int                   dims,
                               global_addr_t         buffer_global_addr,
                               unsigned long long    *buffer_size,
                               data_type_t           dtype);

static void process_shape_stride(
int         *input_shape,
int         *input_stride,
const int   *input_shape_orig,
const int   *input_stride_orig,
int         shape_dim_orig) {
    for(int i = 3; i >= 0; --i) {    // fill 1 in the front dim to make dims=4
        if((i - shape_dim_orig) >= 0) {
            input_shape[3 - i] = 1;
            input_stride[3 - i] = 1;
        }
        else {
            input_shape[i + 4 - shape_dim_orig] = input_shape_orig[i];
            input_stride[i + 4 - shape_dim_orig] = input_stride_orig[i];
        }
    }
    for(int i = 1; i < 4; ++i) {
        if((i - shape_dim_orig) >= 0) {
            input_stride[3 - i] = input_shape[3 - i + 1] * input_stride[3 - i + 1];
        }
    }
    input_stride[0] = input_shape[1] * input_stride[1];
}

void nodechip_permute(
global_addr_t   input_global_addr,
global_addr_t   trans_buffer_global_addr,
global_addr_t   copy_buffer_global_addr,
global_addr_t   output_global_addr,
int             *shape,
int             *stride,
int             *dim_order,
int             dim,
data_type_t     dtype) {
    TPUKERNEL_ASSERT(dim <= 4);
    int cpy_shape_arr[4], cpy_stride_arr[4];
    dim4 cpy_shape, cpy_stride;
    if(dim < 4) {
        process_shape_stride(cpy_shape_arr, cpy_stride_arr, shape, stride, dim);
        cpy_shape.n = cpy_shape_arr[0];
        cpy_shape.c = cpy_shape_arr[1];
        cpy_shape.h = cpy_shape_arr[2];
        cpy_shape.w = cpy_shape_arr[3];
        cpy_stride.n = cpy_stride_arr[0];
        cpy_stride.c = cpy_stride_arr[1];
        cpy_stride.h = cpy_stride_arr[2];
        cpy_stride.w = cpy_stride_arr[3];
    }
    else {
        cpy_shape.n = shape[0];
        cpy_shape.c = shape[1];
        cpy_shape.h = shape[2];
        cpy_shape.w = shape[3];
        cpy_stride.n = stride[0];
        cpy_stride.c = stride[1];
        cpy_stride.h = stride[2];
        cpy_stride.w = stride[3];
    }

    tpu_gdma_cpy_S2S(copy_buffer_global_addr,
                     input_global_addr,
                     &cpy_shape,
                     NULL,
                     &cpy_stride,
                     dtype);
    nodechip_transpose(copy_buffer_global_addr,
                       output_global_addr,
                       shape,
                       dim_order,
                       dim,
                       trans_buffer_global_addr,
                       NULL,
                       dtype);
}
void tpu_kernel_api_permute(const void *args) {
    sg_api_permute_t *api = (sg_api_permute_t*)args;
    
    tpu_initialize();
    nodechip_permute(api->input_global_addr,
                     api->trans_buffer_global_addr,
                     api->copy_buffer_global_addr,
                     api->output_global_addr,
                     api->shape,
                     api->stride,
                     api->dim_order,
                     api->dim,
                     api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_permute);