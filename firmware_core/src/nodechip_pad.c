#include <string.h>

#include "sg_api_struct.h"
#include "tpu_kernel.h"

static inline bool get_is_true3d(int group_type, int dims)
{
    return group_type == 1 && dims > 4;
}

static inline void parse_NCDHW(int group_type, const int *shape, int dims, dim5* out_shape)
{
    out_shape->n = dims > 0 ? shape[0] : 1;
    out_shape->c = dims > 1 ? shape[1] : 1;
    const int is_true3d = get_is_true3d(group_type, dims);
    out_shape->d = is_true3d ? shape[2] : 1;
    out_shape->h = is_true3d ? shape[3] : (dims>2 ? shape[2] : 1);
    out_shape->w = 1;
    for(int i = is_true3d ? 4 : 3; i < dims; i++)
        out_shape->w *= shape[i];
}

static inline void parse_NCHW(const int *shape, int dims, dim4* out_shape)
{
    dim5 out_shape_5d;
    parse_NCDHW(0, shape, dims, &out_shape_5d);
    tpu_local_shape_5d_to_4d(&out_shape_5d, out_shape);
}

extern void nodechip_pad(global_addr_t input_global_addr, global_addr_t output_global_addr,
                         int input_n, int input_c, int input_h, int input_w, int pad[4][2],
                         int type, float constant, data_type_t dtype);

void tpu_kernel_api_pad(const void *args) {
    sg_api_pad_t *api = (sg_api_pad_t*)args;
    int pad[4][2] = {0};
    int offset = api->dim - api->pad_size / 2;
    memcpy(pad + offset, api->pad, api->pad_size * sizeof(int));

    TPUKERNEL_ASSERT(api->pad_size <= 8);
    nodechip_pad(api->input_global_addr, api->output_global_addr, api->shape[0],
                 api->shape[1], api->shape[2], api->shape[3], pad, api->mode, api->value, api->dtype);
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pad);