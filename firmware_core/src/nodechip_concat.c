#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

extern void nodechip_concat_nd (
global_addr_t* input_global_addrs,
global_addr_t output_global_addr,
int ( *input_shapes ) [FW_MAX_SHAPE_DIMS],
int* st_by_concatway,
int dims,
int input_num,
int concat_axis,
data_type_t dtype );

void tpu_kernel_api_concat ( const void * args )
{
  sg_api_concat_t * api = ( sg_api_concat_t * ) args;
  int st_by_concatway[FW_MAX_CONCAT_NUM] = { 0 };
  tpu_initialize();
  nodechip_concat_nd ( api->input_global_addrs,
                       api->output_global_addr,
                       api->input_shapes,
                       st_by_concatway,
                       api->dim,
                       api->input_num,
                       api->axis,
                       ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_concat );

#ifdef FIRMWARE_BACKEND_2260
static inline int count_numel(const int* shape, int beg_axis, int end_axis) {
  int count = 1;
  for (int i = beg_axis; i < end_axis; ++i)
    count *= shape[i];
  return count;
}

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

void nodechip_concat_nd_multi_core(
    global_addr_t* input_global_addrs,
    global_addr_t output_global_addr,
    int (*input_shapes)[FW_MAX_SHAPE_DIMS],
    int* st_by_concatway,
    int dims,
    int input_num,
    int concat_axis,
    data_type_t dtype)
{
    const int core_idx = tpu_core_index();

    if (concat_axis < 0) concat_axis += dims;
    TPUKERNEL_ASSERT(concat_axis >= 0 && concat_axis < dims);
    TPUKERNEL_ASSERT(dims <= FW_MAX_SHAPE_DIMS);
    int outer_num_ = count_numel(input_shapes[0], 0, concat_axis);
    TPUKERNEL_ASSERT(outer_num_ <= INT32_MAX);
    const int outer_num = (int)outer_num_;

    int new_dims = 2;
    int input_shapes_new_shape[input_num][FW_MAX_SHAPE_DIMS];
    int outer_num_real = 1, outer_num_avg =1;
    int min_cores_needed = 1;
    int output_inner_num = 0;
    compute_current_slice_info_multi_core(outer_num, &outer_num_real, &outer_num_avg, &min_cores_needed);
    if (core_idx < min_cores_needed) {
      for (int i = 0; i < input_num; ++i) {
          int input_inner_num_ = count_numel(input_shapes[i], concat_axis, dims);
          TPUKERNEL_ASSERT(input_inner_num_ <= INT32_MAX);
          input_shapes_new_shape[i][0] = outer_num_real;
          input_shapes_new_shape[i][1] = (int)input_inner_num_;
          for (int j = new_dims; j < FW_MAX_SHAPE_DIMS; j++)
            input_shapes_new_shape[i][j] = 1;
          output_inner_num += input_shapes_new_shape[i][1];
      }

      global_addr_t input_global_addrs_new[input_num];
      for (int i = 0; i < input_num; i ++) {
          input_global_addrs_new[i] = input_global_addrs[i] +
                                      core_idx * input_shapes_new_shape[i][1] *
                                      outer_num_avg * tpu_data_type_size(dtype);
      }

      int concat_axis_new = 1;
      if (dims == 1 || concat_axis ==0) concat_axis_new = 0;
      nodechip_concat_nd(
                  input_global_addrs_new,
                  output_global_addr + core_idx * outer_num_avg * output_inner_num * tpu_data_type_size(dtype),
                  input_shapes_new_shape,
                  st_by_concatway,
                  new_dims,
                  input_num,
                  concat_axis_new,
                  dtype);
    }
    tpu_sync_all();
}

void tpu_kernel_api_concat_multi_core ( const void * args )
{
  sg_api_concat_t * api = ( sg_api_concat_t * ) args;
  int st_by_concatway[FW_MAX_CONCAT_NUM] = { 0 };
  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  nodechip_concat_nd_multi_core ( api->input_global_addrs,
                       api->output_global_addr,
                       api->input_shapes,
                       st_by_concatway,
                       api->dim,
                       api->input_num,
                       api->axis,
                      ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_concat_multi_core );
#endif