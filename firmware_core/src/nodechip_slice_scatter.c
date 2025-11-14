#include "sg_api_struct.h"
#include "tpu_kernel.h"


void nodechip_slice_scatter(global_addr_t output_global_addr,
                            global_addr_t input_global_addr,
                            global_addr_t indices_global_addr,
                            global_addr_t param_global_addr, int *input_shape,
                            int param_h, data_type_t dtype) {
  const dim4 shape = {.n = input_shape[0],
                      .c = input_shape[1],
                      .h = input_shape[2],
                      .w = input_shape[3]};
  tpu_gdma_cpy_S2S(output_global_addr, input_global_addr, &shape, NULL, NULL,
                   dtype);
  tpu_gdma_h_scatter_S2S(output_global_addr, param_global_addr,
                         indices_global_addr, false, &shape, param_h, NULL,
                         NULL, NULL, dtype);
}


int tpu_kernel_api_slice_scatter_multi_core(const void *args) {
  sg_api_slice_scatter_t *api = (sg_api_slice_scatter_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);
  int input_shape[4] = {1, 1, 1, 1};
  for (int i = 0; i < api->dim; i++) {
    input_shape[1] *= api->input_shape[i];
  }
  input_shape[2] *= api->input_shape[api->dim];
  for (int i = api->dim + 1; i < api->input_dim; i++) {
    input_shape[3] *= api->input_shape[i];
  }
  #ifdef ENABLE_MULTI_CORE
  tpu_initialize();
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length = input_shape[1];
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1) {
    cur_length_slice = length - length_slice * (length_secs - 1);
  }
  int shape[4] = {1, cur_length_slice, input_shape[2], input_shape[3]};

  int offset = core_idx * length_slice * input_shape[2] * input_shape[3];
  int src_offset =
      core_idx * length_slice * api->src_shape[api->dim] * input_shape[3];

  int dsize = tpu_data_type_size(api->dtype);
  if (core_idx * length_slice < length) {
    nodechip_slice_scatter(api->output_global_addr + offset * dsize,
                           api->input_global_addr + offset * dsize,
                           api->indices_global_addr,
                           api->src_global_addr + src_offset * dsize, shape,
                           api->src_shape[api->dim], api->dtype);
  }
  tpu_poll();
  return 0;
  #else
  nodechip_slice_scatter(api->output_global_addr, api->input_global_addr,
                        api->indices_global_addr, api->src_global_addr,
                        input_shape, api->src_shape[api->dim], api->dtype);
  tpu_poll();
  return 0;
  #endif
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_slice_scatter_multi_core);
