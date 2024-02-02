#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_cast (
global_addr_t   in_global_addr,
global_addr_t   out_global_addr,
const int*      shape,
int             shape_dim,
data_type_t     src_dtype,
data_type_t     dst_dtype,
rounding_mode_t round_mode );

void tpu_kernel_api_dtype_convert ( const void* args ) {
  sg_api_dtype_convert_t* api = ( sg_api_dtype_convert_t* ) args;
  tpu_initialize();
  nodechip_cast (
  api->input_global_addr,
  api->output_global_addr,
  api->shape,
  api->dim,
  ( data_type_t ) api->input_dtype,
  ( data_type_t ) api->output_dtype,
  RM_HALF_TO_EVEN );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_dtype_convert );

#ifdef BACKEND_SG2260
static inline void compute_current_slice_info_multi_core(int total_num, int* expected_current_slice,
                                                         int* expected_avg_slice, int* expected_secs) {
  const int core_num = tpu_core_num();
  const int core_idx = tpu_core_index();
  const int avgnum_element_each_core = DIV_UP(total_num, core_num);
  const int num_max_core_needed = DIV_UP(total_num, avgnum_element_each_core);
  TPUKERNEL_ASSERT(num_max_core_needed <= core_num);
  int current_num_for_current_core = avgnum_element_each_core;
  if (core_idx == num_max_core_needed - 1) {
    current_num_for_current_core = total_num - avgnum_element_each_core * (num_max_core_needed - 1);
  }
  *expected_current_slice = current_num_for_current_core;
  *expected_avg_slice = avgnum_element_each_core;
  *expected_secs = num_max_core_needed;
}

void tpu_kernel_api_dtype_convert_multi_core ( const void* args ) {
  sg_api_dtype_convert_t* api = ( sg_api_dtype_convert_t* ) args;
  tpu_initialize();
#ifdef USING_PERF_MODE
  tpu_sync_all();
#endif
  const int core_num = tpu_core_num();
  const int core_idx = tpu_core_index();
  int new_dim = api->dim >= 2 ? 2 : 1;
  int new_shape[2];
  int outer_num = 1, inner_num = 1;
  if (api->dim > 0 &&  new_dim == 1) {
    new_shape[0] = api->shape[0];
    new_shape[1] = 1;
    outer_num = api->shape[0];
  } else if (api->dim > 0 && new_dim == 2) {
    int split_dim =1;
    for (int i=0; i <api->dim; i++) {
      outer_num *= api->shape[i];
      if (outer_num > core_num) break;
      split_dim += 1;
    }
    for (int i=split_dim; i <api->dim; i++) {
      inner_num *= api->shape[i];
    }
    new_shape[0] = outer_num;
    new_shape[1] = inner_num;
  } else {
    new_shape[0] = 1;
    new_dim = 1;
  }
  int outer_num_real = 1, outer_num_avg =1;
  int min_cores_needed = 1;
  if (api->dim > 0) {
    compute_current_slice_info_multi_core(outer_num, &outer_num_real, &outer_num_avg, &min_cores_needed);
    new_shape[0] = outer_num_real;
  } 
  if (core_idx < min_cores_needed) {
    nodechip_cast (
      api->input_global_addr + core_idx * outer_num_avg * inner_num *  tpu_data_type_size(api->input_dtype),
      api->output_global_addr + core_idx * outer_num_avg * inner_num *  tpu_data_type_size(api->output_dtype),
      new_shape,
      new_dim,
      ( data_type_t ) api->input_dtype,
      ( data_type_t ) api->output_dtype,
      RM_HALF_TO_EVEN );
  }
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_dtype_convert_multi_core );
#endif