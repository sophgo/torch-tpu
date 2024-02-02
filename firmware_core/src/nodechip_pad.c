#include <string.h>

#include "sg_api_struct.h"
#include "tpu_kernel.h"


#define CONSTANT (0)
#define REFLECT (1)
#define SYMMETRIC (2)
#define REPLICATE 3
#define CIRCULAR (4)

extern void nodechip_pad(global_addr_t input_global_addr,
                         global_addr_t output_global_addr, int input_n,
                         int input_c, int input_h, int input_w, int pad[4][2],
                         int type, float constant, data_type_t dtype);

void nodechip_pad_3d(global_addr_t input_global_addr,
                     global_addr_t output_global_addr,
                     global_addr_t buffer_global_addr, int input_n, int input_c,
                     int input_d, int input_h, int input_w, int pad[5][2],
                     int pad_type, float constant, data_type_t dtype) {
  int pad_dim4[4][2] = {0};
  for (int i = 2; i < 4; i++) {
    pad_dim4[i][0] = pad[i + 1][0];
    pad_dim4[i][1] = pad[i + 1][1];
  }
  nodechip_pad(input_global_addr, buffer_global_addr, input_n * input_c,
               input_d, input_h, input_w, pad_dim4, pad_type, constant, dtype);

  int dtype_size = tpu_data_type_size(dtype);
  switch (pad_type) {
    case REPLICATE:

    {
      dim4 shape = {.n = 1,
                    .c = 1,
                    .h = input_h + pad[3][0] + pad[3][1],
                    .w = input_w + pad[4][0] + pad[4][1]};
      dim4 src_shape = {.n = 1,
                        .c = input_d,
                        .h = input_h + pad[3][0] + pad[3][1],
                        .w = input_w + pad[4][0] + pad[4][1]};
      int src_offset = 0, dst_offset = 0;
      int shape_offset = shape.n * shape.c * shape.h * shape.w * dtype_size;
      int src_shape_offset =
          src_shape.n * src_shape.c * src_shape.h * src_shape.w * dtype_size;
      for (int i = 0; i < input_n * input_c; i++) {
        if (pad[2][0] > 0) {
          for (int j = 0; j < pad[2][0]; j++) {
            tpu_gdma_cpy_S2S(output_global_addr + dst_offset,
                             buffer_global_addr + src_offset, &shape, NULL,
                             NULL, dtype);
            dst_offset += shape_offset;
          }
        }

        tpu_gdma_cpy_S2S(output_global_addr + dst_offset,
                         buffer_global_addr + src_offset, &src_shape, NULL,
                         NULL, dtype);
        dst_offset += src_shape_offset;
        src_offset += src_shape_offset - shape_offset;

        if (pad[2][1] > 0) {
          for (int j = 0; j < pad[2][1]; j++) {
            tpu_gdma_cpy_S2S(output_global_addr + dst_offset,
                             buffer_global_addr + src_offset, &shape, NULL,
                             NULL, dtype);
            dst_offset += shape_offset;
          }
        }
        src_offset += shape_offset;
      }
    } break;
  }
}

void tpu_kernel_api_pad(const void *args) {
  sg_api_pad_t *api = (sg_api_pad_t *)args;
  if (api->pad3d) {
    int pad[5][2] = {0};
    int offset = api->dim - api->pad_size / 2;
    memcpy(pad + offset, api->pad, api->pad_size * sizeof(int));
    TPUKERNEL_ASSERT(api->pad_size == 6);
    nodechip_pad_3d(api->input_global_addr, api->output_global_addr,
                    api->buffer_global_addr, api->shape[0], api->shape[1],
                    api->shape[2], api->shape[3], api->shape[4], pad, api->mode,
                    api->value, api->dtype);
  } else {
    int pad[4][2] = {0};
    int offset = api->dim - api->pad_size / 2;
    memcpy(pad + offset, api->pad, api->pad_size * sizeof(int));
    TPUKERNEL_ASSERT(api->pad_size <= 8);
    nodechip_pad(api->input_global_addr, api->output_global_addr, api->shape[0],
                 api->shape[1], api->shape[2], api->shape[3], pad, api->mode,
                 api->value, api->dtype);
  }
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pad);

#ifdef BACKEND_SG2260
void tpu_kernel_api_pad_multi_core(const void *args) {
  sg_api_pad_t *api = (sg_api_pad_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16 || api->dtype == DT_INT32);

  if (api->pad3d) {
    TPUKERNEL_ASSERT(api->pad_size == 6);
    int pad[5][2] = {0};
    int offset = api->dim - api->pad_size / 2;
    memcpy(pad + offset, api->pad, api->pad_size * sizeof(int));
    tpu_initialize();
    int core_num = tpu_core_num();
    int core_idx = tpu_core_index();
    int length = api->shape[0] * api->shape[1];
    int length_slice = DIV_UP(length, core_num);
    int length_secs = DIV_UP(length, length_slice);
    TPUKERNEL_ASSERT(length_secs <= core_num);
    int cur_length_slice = length_slice;
    if (core_idx == length_secs - 1) {
      cur_length_slice = length - length_slice * (length_secs - 1);
    }
    int shape[5] = {1, cur_length_slice, api->shape[2], api->shape[3],
                    api->shape[4]};
    int output_shape[5] = {1, cur_length_slice,
                           api->shape[2] + pad[2][0] + pad[2][1],
                           api->shape[3] + pad[3][0] + pad[3][1],
                           api->shape[4] + pad[4][0] + pad[4][1]};
    int dsize = tpu_data_type_size(api->dtype);
    int input_offset =
        core_idx * length_slice * shape[2] * shape[3] * shape[4] * dsize;
    int output_offset = core_idx * length_slice * output_shape[2] *
                        output_shape[3] * output_shape[4] * dsize;
    if (core_idx * length_slice < length) {
      nodechip_pad_3d(api->input_global_addr + input_offset,
                      api->output_global_addr + output_offset,
                      api->buffer_global_addr + output_offset, shape[0],
                      shape[1], shape[2], shape[3], shape[4], pad, api->mode,
                      api->value, api->dtype);
    }
    tpu_poll();
  } else {
    int pad[4][2] = {0};
    int offset = api->dim - api->pad_size / 2;
    memcpy(pad + offset, api->pad, api->pad_size * sizeof(int));
    tpu_initialize();
    int core_num = tpu_core_num();
    int core_idx = tpu_core_index();
    int length = 1;
    
    length = api->shape[0];
    int length_slice = DIV_UP(length, core_num);
    if (offset == 0){
      length_slice = api->shape[0];
    }
    int length_secs = DIV_UP(length, length_slice);
    TPUKERNEL_ASSERT(length_secs <= core_num);
    int cur_length_slice = length_slice;
    if (core_idx == length_secs - 1) {
      cur_length_slice = length - length_slice * (length_secs - 1);
    }
    int shape[4] = {cur_length_slice, api->shape[1], api->shape[2], api->shape[3]};
    int output_shape[4] = {cur_length_slice + pad[0][0] + pad[0][1], cur_length_slice+ pad[1][0] + pad[1][1],
                           api->shape[2] + pad[2][0] + pad[2][1],
                           api->shape[3] + pad[3][0] + pad[3][1]};
    int dsize = tpu_data_type_size(api->dtype);
    int input_offset = core_idx * length_slice * shape[1] *shape[2] * shape[3] * dsize;
    int output_offset =
        core_idx * length_slice * output_shape[1] *output_shape[2] * output_shape[3] * dsize;
    if (core_idx * length_slice < length) {
      nodechip_pad(api->input_global_addr + input_offset,
                   api->output_global_addr + output_offset, shape[0], shape[1],
                   shape[2], shape[3], pad, api->mode, api->value, api->dtype);
    }
    tpu_poll();
  }
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_pad_multi_core);
#endif