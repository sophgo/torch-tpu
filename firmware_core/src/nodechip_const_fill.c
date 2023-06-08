#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"

void nodechip_fill(
    global_addr_t out_global_addr,
    const int* shape,
    int shape_dim,
    unsigned int filled_value,
    data_type_t filled_dtype)
{
  int shape4d[4] = {1, 1, 1, 1};
  for (int i = 0; i < MIN(shape_dim, 4); i++){
    shape4d[i] = shape[i];
  }
  unsigned long long prod = shape4d[3];
  for(int i = 4; i < shape_dim; i++){
    prod *= shape[i];
  }
  TPUKERNEL_ASSERT(prod <= __INT32_MAX__);
  shape4d[3] = (int)prod;
  scalar_t C = {.u32 = filled_value};
  tpu_gdma_set_C_system(
    out_global_addr,
    C,
    (dim4*)shape4d,
    NULL,
    filled_dtype
  );
}


void tpu_kernel_api_const_fill ( const void * args ){
    sg_api_constant_fill_t *api = (sg_api_constant_fill_t*)args;
    tpu_initialize();
    nodechip_fill(
        api->out_global_addr,
        api->shape,
        api->shape_dim,
        api->filled_value,
        tpu_type_convert(api->filled_sgdtype));
    tpu_poll();

}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_const_fill);