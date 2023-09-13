#include "sg_api_struct.h"
#include "tpu_kernel.h"

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

void tpu_kernel_api_nonzero(const void *args) {
  sg_api_nonzero_t *api = (sg_api_nonzero_t *)args;
  tpu_initialize();
  nodechip_nonzero(api->input_global_addr, api->output_global_addr,
                   api->index_global_addr, api->num_global_addr, api->shape, api->dim,
                   (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_nonzero);