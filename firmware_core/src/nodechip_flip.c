#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_reverse(global_addr_t bottom_global_offset,
                             global_addr_t top_global_offset,
                             int *input_tensor_shape, int dims, int axis,
                             data_type_t dtype);

int tpu_kernel_api_flip_multi_core(const void *args) {
#ifdef BACKEND_SG2260
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
  return 0;
#else
  sg_api_flip_t *api = (sg_api_flip_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_reverse(api->input_global_addr, api->output_global_addr, api->shape,
                   api->dim, api->axis, api->dtype);
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_flip_multi_core);
