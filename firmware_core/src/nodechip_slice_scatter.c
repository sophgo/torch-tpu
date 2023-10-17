#include <string.h>

#include "sg_api_struct.h"
#include "tpu_kernel.h"



void nodechip_slice_scatter(
  global_addr_t output_global_addr, 
    global_addr_t input_global_addr, 
    global_addr_t indices_global_addr,
    global_addr_t param_global_addr,
    int* input_shape,
    int param_h,
    data_type_t dtype
){
  const dim4 shape = {.n = input_shape[0], .c = input_shape[1], .h = input_shape[2], .w = input_shape[3]};
  tpu_gdma_cpy_S2S(output_global_addr,input_global_addr,&shape,NULL,NULL,dtype);
  tpu_gdma_h_scatter_S2S(output_global_addr,param_global_addr,indices_global_addr,false,&shape,param_h,NULL,NULL,NULL,dtype);
}


void tpu_kernel_api_slice_scatter(const void *args) {
  sg_api_slice_scatter_t *api = (sg_api_slice_scatter_t *)args;
  int input_shape[4] = {1,1,1,1};

  for (int i = 0; i < api->dim; i++) {
    input_shape[1] *= api->input_shape[i];
  }
  input_shape[2] *= api->input_shape[api->dim];
  for (int i = api->dim + 1; i < api->input_dim; i++) {
    input_shape[3] *= api->input_shape[i];
  }
  nodechip_slice_scatter(api->output_global_addr,api->input_global_addr,api->indices_global_addr,api->src_global_addr,input_shape,api->src_shape[api->dim],api->dtype);
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_slice_scatter);