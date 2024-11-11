#include <string.h>
#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "nodechip_utils.h"

extern void nodechip_reduce_full(
    global_addr_t input_global_addr, global_addr_t buffer_global_addr,
    global_addr_t output_global_addr, const int *input_shape_orig,
    const int *axis_list_orig, int shape_dims_orig, int axis_num, int method,
    unsigned long long *buffer_size, data_type_t dtype);

int tpu_kernel_api_reduce_max_or_min_multi_core(const void *args) {
  sg_api_reduce_max_or_min_t *api = (sg_api_reduce_max_or_min_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  TPUKERNEL_ASSERT(api->mode == 0 || api->mode == 1);
  int mode = api->mode == 0 ? REDUCE_MAX : REDUCE_MIN;

  if(api->dim == api->reduction_dim_length) {   //reshape
      uint64_t all_element_nums = 1;
      for (int i = 0; i < api->dim; ++i) {
          all_element_nums *= (unsigned int)api->shape[i];
      }
      int vecDivisors[200] = {0};
      int nums =findAllDivisors(all_element_nums, vecDivisors, sizeof(vecDivisors)/sizeof(int));
      int idx = findFactorsIndex(NPU_NUM, vecDivisors);
      api->shape[0] = vecDivisors[idx];
      uint64_t remain_elememts = all_element_nums/vecDivisors[idx];
      memset(vecDivisors, 0, nums * sizeof(int));
      findAllDivisors(remain_elememts, vecDivisors, sizeof(vecDivisors)/sizeof(int));
      idx = findFactorsIndex(EU_NUM, vecDivisors);
      api->shape[1] = remain_elememts / vecDivisors[idx];
      api->shape[2] = vecDivisors[idx];
      api->reduction_dim_length = 3;
      api->dim = api->reduction_dim_length;
      api->reduction_dim[api->reduction_dim_length-1] = api->reduction_dim_length -1;
  }

#ifdef BACKEND_SG2260
  int length = 1;
  if (api->dim > 0) {
    length = api->shape[0];
  }

  tpu_initialize();
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  if (api->reduction_dim_length > 0 && api->reduction_dim[0] == 0) {
    length_slice = api->shape[0];
  }
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }
  int input_offset = length_slice;
  int output_offset = length_slice;

  int shape[FW_MAX_SHAPE_DIMS];
  shape[0] = cur_length_slice;

  for (int i = 1, j = 0; i < api->dim; i++) {
    shape[i] = api->shape[i];
    input_offset *= shape[i];
    if (i == api->reduction_dim[j] && j < api->reduction_dim_length) {
      output_offset *= 1;
      j++;
    } else {
      output_offset *= shape[i];
    }
  }
  int dsize = tpu_data_type_size(api->dtype);
  if (core_idx * length_slice < length) {
    nodechip_reduce_full(
        api->input_global_addr + core_idx * input_offset * dsize,
        api->buffer_global_addr + core_idx * input_offset * dsize,
        api->output_global_addr + core_idx * output_offset * dsize, shape,
        api->reduction_dim, api->dim, api->reduction_dim_length, mode, NULL,
        api->dtype);
  }
  tpu_poll();
  return 0;
#else
  tpu_initialize();
  nodechip_reduce_full(api->input_global_addr, api->buffer_global_addr,
                       api->output_global_addr, api->shape, api->reduction_dim,
                       api->dim, api->reduction_dim_length, mode, NULL,
                       api->dtype);
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_reduce_max_or_min_multi_core);