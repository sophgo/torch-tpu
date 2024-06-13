#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_arg_Nd(global_addr_t input_global_addr,
                            global_addr_t index_global_addr,
                            global_addr_t value_global_addr, const int *shape,
                            int dims, int axis,
                            int method,  // 0: argmax, 1: argmin
                            int is_index_int32, int select_last_index,
                            int need_val, data_type_t dtype);

void nodechip_arg(global_addr_t input_global_addr,
                  global_addr_t buffer_global_addr,
                  global_addr_t values_global_addr,
                  global_addr_t indices_global_addr, int *shape, int dim,
                  int axis, int method, data_type_t dtype) {
  nodechip_arg_Nd(input_global_addr, buffer_global_addr, values_global_addr,
                  shape, dim, axis, method % 2, 1, 0, method >= 2, dtype);
  dim4 shape_dim4 = {.n = 1, .c = shape[1], .h = 1, .w = shape[3]};

  dim4 stride = {.n = shape_dim4.c * shape_dim4.h * shape_dim4.w * 2,
                 .c = shape_dim4.h * shape_dim4.w * 2,
                 .h = shape_dim4.w * 2,
                 .w = 2};
  // the indices should be int64, We can use tpu_gdma_cpy_S2S with stride = 2 to
  // convert int32 to int64
  tpu_gdma_cpy_S2S(indices_global_addr, buffer_global_addr, &shape_dim4,
                   &stride, NULL, DT_INT32);
  scalar_t zero_C = {.u32 = 0};
  tpu_gdma_set_C_system(indices_global_addr + tpu_data_type_size(DT_INT32),
                        zero_C, &shape_dim4, &stride, DT_INT32);
}

int tpu_kernel_api_arg(const void *args) {
  sg_api_reduce_arg_t *api = (sg_api_reduce_arg_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  tpu_initialize();
  nodechip_arg(api->input_global_addr, api->buffer_global_addr,
               api->values_global_addr, api->indices_global_addr, api->shape,
               api->dim, api->axis, api->mode, api->dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_arg);

#ifdef BACKEND_SG2260
int tpu_kernel_api_arg_muti_core(const void *args) {
  sg_api_reduce_arg_t *api = (sg_api_reduce_arg_t *)args;
  // TPUKERNEL_ASSERT(api->dtype == DT_FP32);
  tpu_initialize();

  unsigned int slice_num = tpu_core_num();
  unsigned int slice_idx = tpu_core_index();
  // determine the length of each slice
  TPUKERNEL_ASSERT(slice_num > 0);
  TPUKERNEL_ASSERT(0 <= slice_idx && slice_idx < slice_num);
  unsigned long long length = api->shape[1];
  unsigned long long slice = length > slice_num ? DIV_UP(length, slice_num) : 1;

  // determine the shape of each slice
  int shape[4] = {api->shape[0], slice, api->shape[2], api->shape[3]};
  shape[1] = MIN(slice, length - slice_idx * slice);
  // determine the offset of input and output on each core
  unsigned int input_offset = slice_idx * slice * shape[2] * shape[3];
  unsigned int output_offset = slice_idx * slice * 1 * shape[3];

  if (slice_idx * slice < length) {
    const int dsize = tpu_data_type_size(api->dtype);
    const int int32_size = tpu_data_type_size(DT_INT32);
    const int int64_size = 2 * int32_size;
    nodechip_arg(api->input_global_addr + input_offset * dsize,
                 api->buffer_global_addr + output_offset * int32_size,
                 api->values_global_addr + output_offset * dsize,
                 api->indices_global_addr + output_offset * int64_size, shape,
                 api->dim, api->axis, api->mode, api->dtype);
  }
  tpu_poll();
  return 0;
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_arg_muti_core);
#endif