#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

/*
 * output = nonzero(input)
 */

void nodechip_nonzero(global_addr_t input_global_addr,
                      global_addr_t output_global_addr,
                      global_addr_t index_global_addr,
                      global_addr_t num_global_addr, 
                      int shape[],
                      int dim,
                      data_type_t dtype) {

  dim4 gdma_shape = {.n=1, .c=1, .h=1, .w=1};
  int offset = 0;
  // fuse adjacent dimension
  for (int i=0; i<4; i++){
    if(i >= dim) {break;}
    switch (i){
      case(0) : {
        if (i < 8-dim){
          gdma_shape.n = shape[offset];
          offset += 1;
        } else {
          gdma_shape.n = shape[offset] * shape[offset+1];
          offset += 2;
        }
        break;
      }
      case(1) : {
        if (i < 8-dim){
          gdma_shape.c = shape[offset];
          offset += 1;
        } else {
          gdma_shape.c = shape[offset] * shape[offset+1];
          offset += 2;
        }
        break;
      }
      case(2) : {
        if (i < 8-dim){
          gdma_shape.h = shape[offset];
          offset += 1;
        } else {
          gdma_shape.h = shape[offset] * shape[offset+1];
          offset += 2;
        }
        break;
      }
      case(3) : {
        if (i < 8-dim){
          gdma_shape.w = shape[offset];
          offset += 1;
        } else {
          gdma_shape.w = shape[offset] * shape[offset+1];
          offset += 2;
        }
        break;
      }
    }
  }

  tpu_gdma_nonzero_S2S(index_global_addr, input_global_addr, &gdma_shape, dtype, 0);
  int *num_ptr = (int *) tpu_global_mem_addr(num_global_addr);
  int num = (int) tpu_gdma_get_filter_num();
  *num_ptr = num;
  tpu_flush_cache(num_global_addr, DIV_UP(sizeof(num), 64) * 64);

  tpu_invalidate_cache(index_global_addr, DIV_UP(sizeof(int)*num, 64) * 64);
  int *index = (int *) tpu_global_mem_addr(index_global_addr);
  int *out = (int *) tpu_global_mem_addr(output_global_addr);

  int size[dim];
  size[dim-1] = 1;
  offset = 0;

  for (int i=dim-2; i>=0; i--){
    size[i] = shape[i+1] * size[i+1];
  } 

  int index_temp=0, out_temp=0;
  for (int i=0; i<num; i++){
    index_temp = index[i];
    for (int j=0; j<dim; j++){
      out_temp = index_temp / size[j];
      index_temp -= out_temp * size[j];
      out[offset] = out_temp;
      offset += 1;
    }
  }
  tpu_flush_cache(output_global_addr, DIV_UP( offset*sizeof(out[0]), 64) * 64);
}

void nodechip_reduce_nonzero(
  global_addr_t output_global_addr,
  global_addr_t num_buffer_global_addr,
  global_addr_t num_global_addr,
  int slice,
  int length,
  int dim
) {
  tpu_invalidate_cache(num_buffer_global_addr, 64);
  int *nums = (int *) tpu_global_mem_addr(num_buffer_global_addr);
  int num_total = 0;
  for (int i = 0; i < 8; i++) {
    int cur_num = nums[i * 16];
    if (cur_num > 0) {
      global_addr_t offset = slice * i * length * dim * sizeof(int);
      global_addr_t num_offset = num_total * dim * sizeof(int);
      dim4 now_shape = {1, MIN(NPU_NUM, cur_num), DIV_UP(cur_num, NPU_NUM), dim};
      TPUKERNEL_ASSERT(now_shape.h * dim * 4 < LOCAL_MEM_SIZE);
      tpu_gdma_cpy_S2L(0, output_global_addr + offset, &now_shape, NULL, NULL, DT_INT32);
      dim4 local_shape = now_shape;
      local_shape.w = 1;
      dim4 local_stride = {1, 1, 1, dim};
      tpu_bdc_int_add_C(0, 0, (scalar_t)(slice * i), &local_shape, &local_stride, &local_stride, DT_INT32, DT_INT32, DT_INT32, 0, RM_HALF_AWAY_FROM_ZERO, 1);
      tpu_gdma_cpy_L2S(output_global_addr + num_offset, 0, &now_shape, NULL, NULL, DT_INT32);
      num_total += cur_num;
    }
  }
  dim4 num_shape = {1, 1, 1, 1};
  scalar_t num_scalar = {.u32 = num_total};
  tpu_gdma_set_C_system(num_global_addr, num_scalar, &num_shape, NULL, DT_INT32);
}

void tpu_kernel_api_nonzero(const void *args) {
  sg_api_nonzero_t *api = (sg_api_nonzero_t *)args;
  tpu_initialize();
  nodechip_nonzero(api->input_global_addr, api->output_global_addr,
                   api->index_global_addr, api->num_global_addr, api->shape, api->dim,
                   (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nonzero);

void tpu_kernel_api_nonzero_multicore(const void *args) {
  sg_api_nonzero_multi_core_t *api = (sg_api_nonzero_multi_core_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
  int dsize = tpu_data_type_size(dtype);
  int split_dim = 0;
  for(int i = 0; i < api->dim; ++i) {
    if (api->shape[i] != 1) {
      split_dim = i;
      break;
    }
  }
  tpu_initialize();
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int slice = DIV_UP(api->shape[split_dim], core_num);
  int offset = core_idx * slice;
  int real_slice = MIN(slice, api->shape[split_dim] - offset);
  int real_shape[FW_MAX_SHAPE_DIMS] = {1};
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    if (i == split_dim) {
      real_shape[i] = real_slice;
    } else {
      real_shape[i] = api->shape[i];
      length *= real_shape[i];
    }
  }

  while ((slice * length * dsize) % 64 != 0 && slice < api->shape[split_dim]) {
    ++slice;
    offset = core_idx * slice;
    real_slice = MIN(slice, api->shape[split_dim] - offset);
    real_shape[split_dim] = real_slice;
  }

  if (core_idx == 0) {
    // clear num buffer
    dim4 input_shape = {1, 1, 1, 8};
    dim4 input_stride = {1, 1, 1, 16};
    tpu_gdma_set_C_system(api->num_buffer_global_addr, (scalar_t)0u, &input_shape, &input_stride, DT_INT32);
  }

  tpu_sync_all();
  if (real_slice > 0) {
    nodechip_nonzero(api->input_global_addr + offset * length * dsize,
                      api->output_global_addr + offset * length * sizeof(int) * api->dim,
                      api->index_global_addr + offset * length * sizeof(int),
                      api->num_buffer_global_addr + core_idx * 64,
                      real_shape,
                      api->dim,
                    (data_type_t)api->dtype);
  }
  tpu_sync_all();
  if (core_idx == 0) {
    nodechip_reduce_nonzero(api->output_global_addr,
                            api->num_buffer_global_addr,
                            api->num_global_addr,
                            slice,
                            length,
                            api->dim);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nonzero_multicore);