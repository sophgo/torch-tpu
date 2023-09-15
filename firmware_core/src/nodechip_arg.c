#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern void nodechip_arg_Nd(
    global_addr_t input_global_addr,
    global_addr_t index_global_addr,
    global_addr_t value_global_addr,
    const int*    shape,
    int           dims,
    int           axis,
    int           method, // 0: argmax, 1: argmin
    int           is_index_int32,
    int           select_last_index,
    int           need_val,
    data_type_t   dtype
);

void nodechip_argmax_or_argmin(
    global_addr_t input_global_addr,
    global_addr_t buffer_global_addr,
    global_addr_t values_global_addr,
    global_addr_t indices_global_addr,
    int* shape,
    int dim,
    int axis,
    int method,
    data_type_t dtype
){
    nodechip_arg_Nd(
      input_global_addr,
      buffer_global_addr,
      values_global_addr,
      shape,
      dim,
      axis,
      method%2,
      1,
      0,
      method >= 2,
      dtype
    );
    dim4 shape_dim4 = {.n=1,.c=1,.h=1,.w=1};
    if (axis == dim - 1) {
        for (int i = 0; i < axis; ++i) {
            shape_dim4.c *= shape[i];
        }
    } else {
        for (int i = axis + 1; i < dim; ++i) {
            shape_dim4.w *= shape[i];
        }
        for (int i = 0; i < axis; ++i) {
            shape_dim4.c *= shape[i];
        }
    }
    dim4 stride = {.n = shape_dim4.c * shape_dim4.h * shape_dim4.w * 2,
                   .c = shape_dim4.h * shape_dim4.w * 2,
                   .h = shape_dim4.w * 2,
                   .w = 2 };
    tpu_gdma_cpy_S2S(
      indices_global_addr,
      buffer_global_addr,
      &shape_dim4,
      &stride,
      NULL,
      DT_INT32);
}

void tpu_kernel_api_arg(const void *args) {
  sg_api_reduce_arg_t *api = (sg_api_reduce_arg_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  tpu_initialize();
  nodechip_argmax_or_argmin(
    api->input_global_addr,
    api->buffer_global_addr,
    api->values_global_addr,
    api->indices_global_addr,
    api->shape,
    api->dim,
    api->axis,
    api->mode,
    api->dtype
  );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_arg);