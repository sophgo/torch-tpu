#include "sg_api_struct.h"
#include "tpu_kernel.h"

extern void nodechip_reduce_full(
    global_addr_t input_global_addr, 
    global_addr_t buffer_global_addr,
    global_addr_t output_global_addr, 
    const int *input_shape_orig,
    const int *axis_list_orig, 
    int shape_dims_orig, 
    int axis_num, 
    int method,
    unsigned long long *buffer_size, 
    data_type_t dtype);

void tpu_kernel_api_reduce_max_or_min(const void *args) {
  sg_api_reduce_max_or_min_t *api = (sg_api_reduce_max_or_min_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  TPUKERNEL_ASSERT(api->mode == 0 || api->mode == 1);
  int mode = api->mode == 0 ? REDUCE_MAX : REDUCE_MIN;

  tpu_initialize();
  nodechip_reduce_full( api->input_global_addr, 
                        api->buffer_global_addr,
                        api->output_global_addr,
                        api->shape, 
                        api->reduction_dim,
                        api->dim, 
                        api->reduction_dim_length, 
                        mode, 
                        NULL,
                        api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_reduce_max_or_min);