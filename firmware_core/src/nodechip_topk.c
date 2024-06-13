#include <stdlib.h>


#include "sg_api_struct.h"
#include "tpu_kernel.h"

// void nodechip_topk(global_addr_t input_global_addr,
//                    global_addr_t value_global_addr,
//                    global_addr_t index_global_addr, int *shape, int dim, int
//                    k, int axis, bool largest, bool sorted, int64_t length,
//                    data_type_t dtype) {
//   if (length == 0)
//     return;

//   const int batch_stride = shape[axis];
//   const int batch = length / batch_stride;
//   global_addr_t src_value_addr = input_global_addr;
//   global_addr_t dst_value_addr = value_global_addr;
//   global_addr_t dst_index_addr = index_global_addr;

//   for (int i = 0; i < batch; ++i) {
//     src_value_addr = input_global_addr + i * batch_stride * sizeof(int);
//     dst_value_addr = value_global_addr + i * k * sizeof(int);
//     dst_index_addr = index_global_addr + i * k * sizeof(int);

//     tpu_hau_sort_natural_index(dst_value_addr, dst_index_addr,
//     src_value_addr,
//                                batch_stride, k, largest, dtype);
//     tpu_hau_poll();
//   }
// }

extern void nodechip_batch_topk(system_addr_t bottom_value_addr,
                                system_addr_t bottom_index_addr, // is always int32
                                system_addr_t top_value_addr,
                                system_addr_t top_index_addr, // is always int32
                                bool bottom_index_valid, int k, int descending, int batchs,
                                bool is_batch_same, int *batch_num, int batch_stride,
                                data_type_t dtype);

int tpu_kernel_api_topk(const void *args) {
  sg_api_topk_t *api = (sg_api_topk_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_INT32 ||
                   api->dtype == DT_UINT32);

  int64_t length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  if (api->axis != api->dim - 1) {
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }

  const int batch_stride = api->shape[api->axis];
  const int batchs = length / batch_stride;
  int *batch_nums = (int *)malloc(batchs * sizeof(int));
  for (int i = 0; i < batchs; i++)
    batch_nums[i] = batch_stride;

  tpu_initialize();
  nodechip_batch_topk(api->input_global_addr, (system_addr_t)(-1),
                      api->value_global_addr, api->index_global_addr, false,
                      api->k, api->largest, batchs, true, batch_nums,
                      batch_stride, api->dtype);
  tpu_poll();
  free(batch_nums);
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_topk);

#ifdef BACKEND_SG2260
int tpu_kernel_api_topk_multi_core(const void *args) {
  sg_api_topk_t *api = (sg_api_topk_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_INT32 ||
                   api->dtype == DT_UINT32);

  int64_t length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  if (api->axis != api->dim - 1) {
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }
  const int batch_stride = api->shape[api->axis];
  const int batchs = length / batch_stride;
  int *batch_nums = (int *)malloc(batchs * sizeof(int));
  for (int i = 0; i < batchs; i++)
    batch_nums[i] = batch_stride;

  tpu_initialize();
  int core_idx = tpu_core_index();
  if (core_idx == 0) {
    nodechip_batch_topk(api->input_global_addr, (system_addr_t)(-1),
                        api->value_global_addr, api->index_global_addr, false,
                        api->k, api->largest, batchs, true, batch_nums,
                        batch_stride, api->dtype);
  }
  tpu_poll();
  free(batch_nums);
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_topk_multi_core);
#endif