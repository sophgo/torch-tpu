#include "sg_api_struct.h"
#include "tpu_kernel.h"
inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}


void nodechip_cast_i64_to_i32_without_overflow(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    const int* shape,
    int shape_dim,
    rounding_mode_t round_mode)
{
  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
  local_addr_t in_local_addr[2] = {0, 4 * bank_size};

  unsigned long long length = 1;
  for (int i = 0; i < shape_dim; i++) {
    length *= (unsigned long long)shape[i];
  }

  /// continus 128byte can get better peformance
  data_type_t src_dtype = DT_INT32;
  data_type_t dst_dtype = DT_INT32;
  data_type_t use_dtype = tpu_data_type_size(src_dtype) > tpu_data_type_size(dst_dtype)
                          ? dst_dtype : src_dtype;
  int npu_num = tpu_npu_num();
  int eu_num = tpu_eu_num(use_dtype);
  int tensor_w = MAX(DIV_UP(MIN(length, 32768), npu_num), DIV_UP(128, (unsigned long long)eu_num * tpu_data_type_size(use_dtype)));
  unsigned long long slice = MIN(MIN(length, (unsigned long long)npu_num * tensor_w), 32768);

  int max_rows_per_time = (4*bank_size) / (tensor_w*MAX(tpu_data_type_size(src_dtype), tpu_data_type_size(dst_dtype)));
  int rows = DIV_UP(length, slice);
  int rows_secs = DIV_UP(rows, max_rows_per_time);
  // at least loop two times to overlap all bdc time
  int rows_slice = DIV_UP(rows, MAX(rows_secs, 2));

  unsigned long long cur_idx[2] = {0}, cur_rows[2] = {0}, cur_cols[2] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[1] < length) {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1) {
      unsigned long long cur_len = MIN(length - cur_idx[0], rows_slice * slice);
      cur_cols[0] = MIN(cur_len, slice);
      cur_rows[0] = MAX(1, cur_len / cur_cols[0]); // don't use DIV_UP
    }

    // store output
    if (stage_idx > 0) {
      dim4 top_local_shape = {1, cur_rows[1], 1, cur_cols[1]};
      dim4 top_local_stride;
      tpu_aligned_stride(&top_local_stride, 0, &top_local_shape, dst_dtype);

      dim4 top_global_shape = {1, cur_rows[1], 1, cur_cols[1]};
      dim4 top_global_stride;
      tpu_continuous_stride(&top_global_stride, &top_global_shape);

      tpu_gdma_cpy_L2S(
          out_global_addr + cur_idx[1] * tpu_data_type_size(dst_dtype),
          in_local_addr[(stage_idx-1)&01],
          &top_local_shape,
          &top_global_stride,
          &top_local_stride,
          dst_dtype
      );
      top_global_shape.n = 1;
    }

    // load input
    if (draning_idx < 1) {
      dim4 dst_local_shape = {1, cur_rows[0], 1, cur_cols[0]};
      dim4 dst_local_stride;
      tpu_aligned_stride(&dst_local_stride, 0, &dst_local_shape, dst_dtype);

      dim4 bottom_global_shape = {1, cur_rows[0], 1, cur_cols[0]};
      dim4 src_global_stride;
      src_global_stride.w = 2;
      src_global_stride.h = bottom_global_shape.w;
      src_global_stride.c = bottom_global_shape.h * bottom_global_shape.w;
      src_global_stride.n = bottom_global_shape.c * src_global_stride.c;

      tpu_gdma_general_cpy_S2L(
          in_local_addr[stage_idx&01],
          in_global_addr + cur_idx[0] * tpu_data_type_size(src_dtype),
          &dst_local_shape,
          &bottom_global_shape,
          &dst_local_stride,
          &src_global_stride,
          src_dtype);
      src_global_stride.h = bottom_global_shape.w;
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 2);
    pipeline_move(cur_cols, 2);
    pipeline_move(cur_rows, 2);
    if (draning_idx < 1) {
      cur_idx[0] += cur_cols[0] * cur_rows[0];
      if (cur_idx[0] >= length) {
        draning_idx++;
      }
    } else {
      draning_idx++;
    }
    stage_idx++;
  }
}

extern void nodechip_index_select (
global_addr_t input_global_addr,
global_addr_t index_global_addr,
global_addr_t output_global_addr,
const int     *input_shape,
int           shape_dims,
int           index_num,
int           axis, // axis to do index_select
int           const_val, // fill_value if index not found in input
data_type_t   dtype );

void tpu_kernel_api_index_select ( const void * args )
{
  sg_api_index_select_t *api = ( sg_api_index_select_t * ) args;
  tpu_initialize();
  int input_shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  for ( int i = 0; i < api->dim; ++i ) {
    input_shape[i] = api->input_shape[i];
  }
  // if (api->is_index_int64)  {
  //   int index_shape[2] = {2, api->index_num};
  //   nodechip_cast_i64_to_i32_without_overflow(
  //     api->index_global_addr,
  //     api->index_global_addr,
  //     index_shape,
  //     2,
  //     RM_HALF_TO_EVEN);
  // }
  nodechip_index_select (
  api->input_global_addr,
  api->index_global_addr,
  api->output_global_addr,
  input_shape,
  api->dim,
  api->index_num,
  api->axis,
  0,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_index_select );
