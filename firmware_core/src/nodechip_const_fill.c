#include "sg_api_struct.h"
#include "tpu_kernel.h"

void nodechip_fill (
global_addr_t out_global_addr,
const int* shape,
int shape_dim,
unsigned int filled_value,
data_type_t filled_dtype )
{
  int shape4d[4] = {1, 1, 1, 1};
  for ( int i = 0; i < MIN ( shape_dim, 4 ); i++ ) {
    shape4d[i] = shape[i];
  }
  unsigned long long prod = shape4d[3];
  for ( int i = 4; i < shape_dim; i++ ) {
    prod *= shape[i];
  }
  TPUKERNEL_ASSERT ( prod <= __INT32_MAX__ );
  shape4d[3] = ( int ) prod;
  scalar_t C = {.u32 = filled_value};
  tpu_gdma_set_C_system (
  out_global_addr,
  C,
  ( dim4* ) shape4d,
  NULL,
  filled_dtype
  );
}

void tpu_kernel_api_const_fill ( const void * args ) {
  sg_api_constant_fill_t *api = ( sg_api_constant_fill_t* ) args;
  tpu_initialize();
  nodechip_fill (
  api->output_global_addr,
  api->shape,
  api->dim,
  api->value,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_const_fill );

void nodechip_constant_fill_multi_core(
    global_addr_t out_global_addr,
    const int* shape,
    int shape_dim,
    unsigned int filled_value,
    data_type_t filled_dtype )
{
  const int core_num = tpu_core_num();
  const int core_idx = tpu_core_index();
  unsigned long long total_num = 1;
  for (int i =0; i < shape_dim; i++)
    total_num *= shape[i];
  TPUKERNEL_ASSERT ( total_num <= __INT32_MAX__ );
  scalar_t C = {.u32 = filled_value};

  const int avgnum_element_each_core = DIV_UP(total_num, core_num);
  const int num_max_core_needed = DIV_UP(total_num, avgnum_element_each_core);
  TPUKERNEL_ASSERT(num_max_core_needed <= core_num);
  int current_num_for_current_core = avgnum_element_each_core;
  if (core_idx == num_max_core_needed - 1) {
    current_num_for_current_core = total_num - avgnum_element_each_core * (num_max_core_needed - 1);
  }
  // int shape4d_for_one_specific_core[4] = {1, 1, 1, 1};
  // search_proper_shape_multicore(shape4d_for_one_specific_core, current_num_for_current_core);
  tpu_gdma_system_set(
  out_global_addr + core_idx * avgnum_element_each_core * tpu_data_type_size(filled_dtype),
  C,
  current_num_for_current_core,
  filled_dtype
  );
}

void tpu_kernel_api_const_fill_multi_core ( const void * args ) {
  sg_api_constant_fill_t *api = ( sg_api_constant_fill_t* ) args;
  tpu_initialize();
  nodechip_constant_fill_multi_core (
  api->output_global_addr,
  api->shape,
  api->dim,
  api->value,
  (data_type_t)api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_const_fill_multi_core );