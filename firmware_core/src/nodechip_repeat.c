#include "sg_api_struct.h"
#include "tpu_kernel.h"



void nodechip_repeat(global_addr_t input_global_addr,
                     global_addr_t output_global_addr, int *shape,
                     int *repeat_times, int dim, int repeat_dim,
                     data_type_t dtype) {
  global_addr_t src_global_addr = input_global_addr;
  global_addr_t dst_global_addr = output_global_addr;
  int cur_dim = 0;
  int dtype_size = tpu_data_type_size(dtype);
  if (dim >= 2) {
    cur_dim = shape[dim - 1];
    for (int i = 0; i < repeat_times[repeat_dim - 2]; ++i) { // 重复h维度
      src_global_addr = input_global_addr;
      for (int h = 0; h < shape[dim - 2]; ++h) {
        for (int k = 0; k < repeat_times[repeat_dim - 1]; ++k) { // 先重复w维度
          dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
          tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL,
                           NULL, dtype);
          dst_global_addr += cur_dim * dtype_size;
        }
        src_global_addr += cur_dim * dtype_size;
      }
    }
  } else if (dim >= 1) { // 仅存在w维度
    cur_dim = shape[dim - 1];
    for (int i = 0; i < repeat_times[repeat_dim - 1]; ++i) {
      dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
      tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL, NULL,
                       dtype);
      dst_global_addr += cur_dim * dtype_size;
    }
    src_global_addr += cur_dim * dtype_size;
  }
  if (dim >= 3) { // 基于上面的代码，c维度的第一个channel已经重复完毕
    for (int c = 1; c < shape[dim - 3];
         ++c) { // 所以从c=1开始，与上面对h、w维度的操作一致
      for (int i = 0; i < repeat_times[repeat_dim - 2]; ++i) {
        src_global_addr = input_global_addr +
                          shape[dim - 1] * shape[dim - 2] * dtype_size * c;
        for (int h = 0; h < shape[dim - 2]; ++h) {
          for (int k = 0; k < repeat_times[repeat_dim - 1]; ++k) {
            dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
            tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL,
                             NULL, dtype);
            dst_global_addr += cur_dim * dtype_size;
          }
          src_global_addr += cur_dim * dtype_size;
        }
      }
    }
    src_global_addr = output_global_addr;
    cur_dim = shape[dim - 1] * repeat_times[repeat_dim - 1] * shape[dim - 2] *
              repeat_times[repeat_dim - 2] * shape[dim - 3];
    for (int i = 1; i < repeat_times[repeat_dim - 3]; ++i) { // 重复c维度
      dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
      tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL, NULL,
                       dtype);
      dst_global_addr += cur_dim * dtype_size;
    }
  }
  if (dim >= 4) {
    for (int n = 1; n < shape[dim - 4];
         ++n) { // 基于上面的代码，n维度的第一个batch已经处理完毕，故从n=1开始
      cur_dim = shape[dim - 1];
      src_global_addr = input_global_addr + n * shape[dim - 1] *
                                                shape[dim - 2] *
                                                shape[dim - 3] * dtype_size;
      for (int c = 0; c < shape[dim - 3]; ++c) {
        for (int i = 0; i < repeat_times[repeat_dim - 2]; ++i) {
          for (int h = 0; h < shape[dim - 2]; ++h) {
            for (int k = 0; k < repeat_times[repeat_dim - 1]; ++k) {
              dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
              tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape,
                               NULL, NULL, dtype);
              dst_global_addr += cur_dim * dtype_size;
            }
            src_global_addr += cur_dim * dtype_size;
          }
          src_global_addr -= shape[dim - 1] * shape[dim - 2] * dtype_size;
        }
        src_global_addr += shape[dim - 1] * shape[dim - 2] * dtype_size;
      }
      src_global_addr = output_global_addr +
                        n * shape[dim - 1] * repeat_times[repeat_dim - 1] *
                            shape[dim - 2] * repeat_times[repeat_dim - 2] *
                            shape[dim - 3] * repeat_times[repeat_dim - 3] *
                            dtype_size;
      ;
      cur_dim = shape[dim - 1] * repeat_times[repeat_dim - 1] * shape[dim - 2] *
                repeat_times[repeat_dim - 2] *
                shape[dim - 3]; // 第一个channel的大小
      for (int i = 1; i < repeat_times[repeat_dim - 3]; ++i) {
        dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
        tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL,
                         NULL, dtype);
        dst_global_addr += cur_dim * dtype_size;
      }
    }
    src_global_addr = output_global_addr; // 起始地址
    cur_dim = shape[dim - 1] * repeat_times[repeat_dim - 1] * shape[dim - 2] *
              repeat_times[repeat_dim - 2] * shape[dim - 3] *
              repeat_times[repeat_dim - 3] *
              shape[dim - 4];                         // 第一个batch的大小
    for (int i = 1; i < repeat_times[dim - 4]; ++i) { // 重复n维度
      dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
      tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL, NULL,
                       dtype);
      dst_global_addr += cur_dim * dtype_size;
    }
  }
  if (repeat_dim > dim) {
    int dist = repeat_dim - dim;
    for (int i = dist - 1; i >= 0; --i) {
      cur_dim = 1;
      src_global_addr = output_global_addr;
      for (int j = 0; j < dim; ++j) {
        cur_dim *= shape[j] * repeat_times[j + dist];
      }
      for (int j = i + 1; j < dist; ++j) {
        cur_dim *= repeat_times[j];
      }
      dim4 cur_shape = {.n = 1, .c = 1, .h = 1, .w = cur_dim};
      for (int j = 1; j < repeat_times[i]; ++j) {
        tpu_gdma_cpy_S2S(dst_global_addr, src_global_addr, &cur_shape, NULL,
                         NULL, dtype);
        dst_global_addr += cur_dim * dtype_size;
      }
    }
  }
}

void nodechip_repeat_spec_for_last_dim(global_addr_t input_global_addr,
                     global_addr_t output_global_addr, int rows, int cols,
                     int repeats, data_type_t dtype) {
  // only support repeat last dim
  int dsize = tpu_data_type_size(dtype);
  dim4 src_shape = {.n = 1, .c = 1, .h = rows, .w = cols};
  dim4 dst_stride = {.w = 1, .h = src_shape.w * repeats, .c = src_shape.w * repeats * src_shape.h, .n = src_shape.w * repeats * src_shape.h * src_shape.c};
  for (int i=0; i<repeats; ++i) {
    tpu_gdma_cpy_S2S(output_global_addr + cols * i * dsize, input_global_addr, &src_shape, &dst_stride, NULL, dtype);
  }
}

bool is_last_dim_spec(sg_api_repeat_t *api) {
  if (api->repeat_dim != api->dim)
    return false;
  for (int i = 0; i < api->dim - 1; ++i) {
    if (api->repeat_times[i] != 1)
      return false;
  }
  return true;
}

void tpu_kernel_api_repeat(const void *args) {
  sg_api_repeat_t *api = (sg_api_repeat_t *)args;

  tpu_initialize();
  nodechip_repeat(api->input_global_addr, api->output_global_addr, api->shape,
                  api->repeat_times, api->dim, api->repeat_dim, api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_repeat);

#ifdef BACKEND_SG2260
void tpu_kernel_api_repeat_multi_core(const void *args) {
  sg_api_repeat_t *api = (sg_api_repeat_t *)args;

  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  int core_idx = tpu_core_index();
  int core_num = tpu_core_num();
  int dsize = tpu_data_type_size(api->dtype);
  // multi-core opt only support repeatting last dim now
  if (is_last_dim_spec(api)) {
    int rows = 1, cols = api->shape[api->dim - 1];
    for (int i = 0; i < api->dim - 1; ++i) {
      rows *= api->shape[i];
    }
    int repeats = api->repeat_times[api->repeat_dim - 1];
    int row_sp = DIV_UP(rows, core_num);
    int core_slice = MIN(row_sp, rows - row_sp * core_idx);
    if (core_slice > 0) {
      nodechip_repeat_spec_for_last_dim(
          api->input_global_addr + row_sp * cols * dsize * core_idx,
          api->output_global_addr + row_sp * cols * repeats * dsize * core_idx, core_slice,
          cols, repeats, api->dtype);
    }
  }
  // TODO: other op depends on repeat, just use core 0 to do
  else if (core_idx == 0)
    nodechip_repeat(api->input_global_addr, api->output_global_addr, api->shape,
                    api->repeat_times, api->dim, api->repeat_dim, api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_repeat_multi_core);
#endif