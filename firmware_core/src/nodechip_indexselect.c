#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

void nodechip_cast_i64_to_i32_without_overflow_pipeline(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    const int* shape,
    int shape_dim,
    int index_num_int32_native,
    rounding_mode_t round_mode)
{
  int bank_size =  tpu_local_mem_size_per_npu() / tpu_bank_num();
  local_addr_t in_local_addr[2] = {0, 4 * bank_size};

  unsigned long long length = 1;
  for (int i = 0; i < shape_dim; i++) {
    length *= (unsigned long long)shape[i];
  }
  unsigned long long max_len = 32768;
  unsigned long long best_continoud_mem = 128;
    /// continus 128byte can get better peformance
  data_type_t src_dtype = DT_INT32;
  data_type_t dst_dtype = DT_INT32;
  data_type_t use_dtype = tpu_data_type_size(src_dtype) > tpu_data_type_size(dst_dtype)
                          ? dst_dtype : src_dtype;
  int npu_num = tpu_npu_num();
  int eu_num = tpu_eu_num(use_dtype);
  int tensor_w = MAX(DIV_UP(MIN(length, max_len), npu_num), DIV_UP(best_continoud_mem, (unsigned long long)eu_num * tpu_data_type_size(use_dtype)));
  unsigned long long slice = MIN(MIN(length, (unsigned long long)npu_num * tensor_w), max_len);

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

      // tpu_continuous_stride(&top_global_stride, &top_global_shape);
      top_global_stride.w = 1;
      top_global_stride.h = index_num_int32_native;
      top_global_stride.c = top_global_shape.h* index_num_int32_native;
      top_global_stride.n = top_global_shape.c * top_global_stride.c;

      tpu_gdma_cpy_L2S(
          out_global_addr + cur_idx[1] * tpu_data_type_size(dst_dtype),
          in_local_addr[(stage_idx-1)&01],
          &top_local_shape,
          &top_global_stride,
          &top_local_stride,
          dst_dtype
      );
    }

    // load input
    if (draning_idx < 1) {
      dim4 dst_local_shape = {1, cur_rows[0], 1, cur_cols[0]};
      dim4 dst_local_stride;
      tpu_aligned_stride(&dst_local_stride, 0, &dst_local_shape, dst_dtype);

      dim4 bottom_global_shape = {1, cur_rows[0], 1, cur_cols[0]};
      dim4 src_global_stride;
      src_global_stride.w = 2;
      src_global_stride.h = index_num_int32_native;
      src_global_stride.c = bottom_global_shape.h * index_num_int32_native;
      src_global_stride.n = bottom_global_shape.c * src_global_stride.c;

      tpu_gdma_general_cpy_S2L(
          in_local_addr[stage_idx&01],
          in_global_addr + cur_idx[0] * tpu_data_type_size(src_dtype),
          &dst_local_shape,
          &bottom_global_shape,
          &dst_local_stride,
          &src_global_stride,
          src_dtype);
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

void nodechip_cast_i64_to_i32_without_overflow(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    const int* shape, //[2,indedx_num]
    int shape_dim,
    int index_num_int32_native,
    rounding_mode_t round_mode)
{
      local_addr_t in_local_addr = 0;
      data_type_t src_dtype = DT_INT32;
      data_type_t dst_dtype = DT_INT32;


      dim4 bottom_global_shape = {1, 1, 1, shape[1]};
      dim4 src_global_stride;
      src_global_stride.w = 2;
      src_global_stride.h =  index_num_int32_native;
      src_global_stride.c = 0;
      src_global_stride.n = 0;
      tpu_poll();

      int* tmp3 = NULL;
      tmp3 = ( int* ) tpu_global_mem_addr ( in_global_addr );
      tmp3[0] = tmp3[0];

      tpu_gdma_general_cpy_S2L(
          in_local_addr,
          in_global_addr,
          &bottom_global_shape,
          &bottom_global_shape,
          NULL,
          &src_global_stride,
          src_dtype);

      dim4 top_global_shape = {1,1, 1,shape[1]};
      dim4 top_global_stride;

      // tpu_continuous_stride(&top_global_stride, &top_global_shape);
      top_global_stride.w = 1;
      top_global_stride.h = index_num_int32_native;
      top_global_stride.c = 0;
      top_global_stride.n = 0;

      tpu_gdma_cpy_L2S(
          out_global_addr,
          in_local_addr,
          &top_global_shape,
          &top_global_stride,
          NULL,
          dst_dtype
      );
      tpu_poll();
      int* tmp4 = NULL;
      tmp4 = ( int* ) tpu_global_mem_addr ( out_global_addr );
      tmp4[0] = tmp4[0];
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
  if (api->is_index_int64)  {
    int index_shape[2] = {2, api->index_num};
    nodechip_cast_i64_to_i32_without_overflow_pipeline(
      api->index_global_addr,
      api->index_global_addr,
      index_shape,
      2,
      api->index_num,
      RM_HALF_TO_EVEN);
  }
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


#ifdef FIRMWARE_BACKEND_2260
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

void nodechip_cast_i64_to_i32_without_overflow_multi_core(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    const int* shape, // {2, api->index_num};
    int shape_dim,  //2
    rounding_mode_t round_mode) {

    const int core_idx = tpu_core_index();

    int outer_num_ = shape[1];
    int outer_num_real = 1, outer_num_avg =1;
    int min_cores_needed = 1;
    //1st_level_Slicing intracore
    compute_current_slice_info_multi_core(outer_num_, &outer_num_real, &outer_num_avg, &min_cores_needed);
    if (core_idx < min_cores_needed) {
      //DT_INT64 has not implemented in TPU1686, INT64 represented by index_shape_sliced[0]==2
      // Thus, in_global_addr_sliced just mul_ INT32 not INT64
      global_addr_t in_global_addr_sliced = in_global_addr +  core_idx * outer_num_avg * 2* tpu_data_type_size(DT_INT32);
      global_addr_t out_global_addr_sliced = out_global_addr  + core_idx * outer_num_avg * 2 *  tpu_data_type_size(DT_INT32);
      int index_shape_sliced[2] = {2, outer_num_real};
      nodechip_cast_i64_to_i32_without_overflow(
        in_global_addr_sliced,
        out_global_addr_sliced,
        index_shape_sliced,
        2,
        outer_num_real,
        round_mode
      );
    }
    tpu_sync_all();
    //Incase inplace-input is occupied in such competition case:
    //  {1, 0 ,2 , 0} , {3,0, 4,0}, surely,
    //  Chip 0 {1, 0, 2, 0} -> {1,2,0,0}
    //  Chip 1 {3, 0, 4, 0} -> {?,?,3,4}
    //  But if chip 1 is in higher prioity, chip 0 will be
    //   Chip 0 {1, 0, 3, 4} x-> {1,2,0,0}
    if (core_idx==0 && min_cores_needed>1) {
      for (int i=0; i< min_cores_needed;i++) {
        local_addr_t in_local_addr = 0;
        data_type_t src_dtype = DT_INT32;
        data_type_t dst_dtype = DT_INT32;
        dim4 bottom_global_shape = {1, 1, 1, outer_num_avg};
        //[Warning] Here, enforce max{core_idx} core using outer_num_real<=outer_num_avg;
        if (i==min_cores_needed-1)
          bottom_global_shape.w = outer_num_real;
        dim4 src_global_stride;
        tpu_continuous_stride(&src_global_stride, &bottom_global_shape);
        src_global_stride.w = 1;
        src_global_stride.h = outer_num_;
        src_global_stride.n = 0;
        src_global_stride.c = 0;

        global_addr_t in_global_addr_final = out_global_addr +  i * outer_num_avg * 2* tpu_data_type_size(DT_INT32);
        global_addr_t out_global_addr_final = out_global_addr +  i * outer_num_avg * 1* tpu_data_type_size(DT_INT32);
        tpu_gdma_general_cpy_S2L(
            in_local_addr,
            in_global_addr_final,
            &bottom_global_shape,
            &bottom_global_shape,
            NULL,
            &src_global_stride,
            src_dtype);

        tpu_gdma_cpy_L2S(
            out_global_addr_final,
            in_local_addr,
            &bottom_global_shape,
            &src_global_stride,
            NULL,
            dst_dtype
        );
      }
    }
}

void nodechip_index_select_multi_core_split_inner(  
    global_addr_t input_global_addr,
    global_addr_t index_global_addr,
    global_addr_t output_global_addr,
    const int     outer_num,
    int           inner_num,
    int           select_num,
    int           index_num,
    int           const_val,
    data_type_t   dtype
) {
  TPUKERNEL_ASSERT(outer_num == 1);
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int inner_sliced = DIV_UP(inner_num, core_num);
  int real_inner_sliced = MIN(inner_num - inner_sliced * core_idx, inner_sliced);

  scalar_t const_filler = {.s32 = const_val};
  if (real_inner_sliced > 0) {
    int out_offset = inner_sliced * core_idx * tpu_data_type_size(dtype);
    int in_offset = inner_sliced * core_idx * tpu_data_type_size(dtype);
    dim4 real_oshape = {1, 1, index_num, real_inner_sliced};
    dim4 output_stride = {.n=inner_num, .c=inner_num, .h=inner_num, .w=1};
    dim4 params_stride = {.n=inner_num, .c=inner_num, .h=inner_num, .w=1};
    
    tpu_gdma_h_gather_S2S(
      output_global_addr + out_offset,
      input_global_addr + in_offset,
      index_global_addr,
      false,
      const_filler,
      &real_oshape,
      select_num,
      &output_stride,
      &params_stride,
      NULL,
      dtype);
  }
}

void nodechip_index_select_multi_core_split_index(  
    global_addr_t input_global_addr,
    global_addr_t index_global_addr,
    global_addr_t output_global_addr,
    const int     outer_num,
    int           inner_num,
    int           select_num,
    int           index_num,
    int           const_val,
    data_type_t   dtype
) {
  TPUKERNEL_ASSERT(outer_num == 1);
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int index_sliced = DIV_UP(index_num, core_num);
  int allocated_core = DIV_UP(index_num, index_sliced);
  if (core_idx == allocated_core - 1)
    index_sliced = index_num - core_idx * index_sliced;
 
  scalar_t const_filler = {.s32 = const_val};
  if (core_idx < allocated_core) {
    int out_offset = index_sliced * core_idx * inner_num * tpu_data_type_size(dtype);
    int index_offset = index_sliced * core_idx * tpu_data_type_size(DT_INT32);
    dim4 real_oshape = {1, 1, index_sliced, inner_num};
    
    tpu_gdma_h_gather_S2S(
      output_global_addr + out_offset,
      input_global_addr,
      index_global_addr + index_offset,
      false,
      const_filler,
      &real_oshape,
      select_num,
      NULL,
      NULL,
      NULL,
      dtype);
  }
}

void nodechip_index_select_multi_core(
    global_addr_t input_global_addr,
    global_addr_t index_global_addr,
    global_addr_t output_global_addr,
    const int     *input_shape,
    int           shape_dims,
    int           index_num,
    int           axis, // axis to do index_select
    int           const_val, // fill_value if index not found in input
    data_type_t   dtype) {


  const int core_idx = tpu_core_index();
  dim4 ishape = {1, 1, 1, 1};
  // e.g. input_shape = (a, b, c, d, e)
  // aixs = 0 -> ishape = (1, 1, a, b*c*d*e)
  // aixs = 1 -> ishape = (1, a, b, c*d*e)
  // axis = 2 -> ishape = (1, a*b, c, d*e)
  // axis = 3 -> ishape = (1, a*b*c, d, e)
  // axis = 4 -> ishape = (1, a*b*c*d, e, 1)
  // n == 1 always
  for (int i = axis+1; i < shape_dims; ++i) {
    ishape.w *= input_shape[i];
  }
  ishape.h = input_shape[axis];
  for (int i = axis-1; i >= 0; --i) {
    ishape.c *= input_shape[i];
  }
  //Fast Pattern: slicing ishape.c so that axis shall be kept in ishape.h
  int outer_num_ = ishape.c;
  int outer_num_real = 1, outer_num_avg =1;
  int min_cores_needed = 1;

  //1st_level_Slicing among CPU
  compute_current_slice_info_multi_core(outer_num_, &outer_num_real, &outer_num_avg, &min_cores_needed);

  dim4 index_stride = {
        .n = 0,
        .c = 0,
        .h = 1,
        .w = 1
  };
  scalar_t const_filler = {.s32 = const_val};
  if (core_idx < min_cores_needed ) {
      //update real ishape.c
      ishape.c = outer_num_real;
      //reset oshape
      dim4 oshape = {ishape.n, ishape.c, index_num, ishape.w};
      dim4 input_global_stride, output_global_stride;
      //As ishape.c sliced and ishape.n==1, continuous mem access is kept.
      tpu_continuous_stride(&input_global_stride, &ishape);
      tpu_continuous_stride(&output_global_stride, &oshape);
      int type_size = tpu_data_type_size(dtype);
      //input [1, c_sliced, h ,w ], h*w strides
      //index []
      //output [1, c_sliced, index_num, ishape.w] index_num * ishape.w strides
      int in_offset_multi_core = outer_num_avg * core_idx * type_size * ishape.w *  ishape.h;
      int out_offset_multi_core = outer_num_avg * core_idx * type_size * index_num * ishape.w;
      // int index_offset_multi_core = outer_num_avg * core_idx * type_size * ishape.w ; // index should not be sliced
      int cidx = 0, widx = 0;

      //2nd_level_Slicing inside CPU
      while (cidx < ishape.c) {
        int cslice = MIN(ishape.c - cidx, 65535);
        int wslice = MIN(ishape.w - widx, 65535);

        dim4 real_oshape = {ishape.n, cslice, index_num, wslice};
        int in_offset = (ishape.n * cidx * ishape.h * ishape.w + widx) * type_size;
        int out_offset = (ishape.n * cidx * index_num * ishape.w + widx) * type_size;
        tpu_gdma_h_gather_S2S(
            output_global_addr + out_offset + out_offset_multi_core,
            input_global_addr  + in_offset + in_offset_multi_core,
            index_global_addr,
            false,
            const_filler,
            &real_oshape,
            ishape.h,
            &output_global_stride,
            &input_global_stride,
            &index_stride,
            dtype);

        widx += wslice;
        if (widx >= ishape.w) {
          widx = 0;
          cidx += cslice;
        }
      }
  }
}

void tpu_kernel_api_index_select_multi_core ( const void * args )
{
  sg_api_index_select_t *api = ( sg_api_index_select_t * ) args;
  tpu_initialize();
  int input_shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  for ( int i = 0; i < api->dim; ++i ) {
    input_shape[i] = api->input_shape[i];
  }
  if (api->is_index_int64)  {
    int index_shape[2] = {2, api->index_num};
    nodechip_cast_i64_to_i32_without_overflow_multi_core(
      api->index_global_addr,
      api->index_global_addr,
      index_shape,
      2,
      RM_HALF_TO_EVEN);
  }
  tpu_sync_all();

  int outer_num = 1;
  int select_num = 1;
  int inner_num = 1;
  for (int i = api->axis + 1; i < api->dim; ++i) {
    inner_num *= input_shape[i];
  }
  select_num = input_shape[api->axis];
  for (int i = api->axis - 1; i >= 0; --i) {
    outer_num *= input_shape[i];
  }
  // case: [1], [32000, 8192] -> [8192] can not make full use of multi core
  if (outer_num == 1) {
    if (select_num < inner_num) {
      nodechip_index_select_multi_core_split_inner(
          api->input_global_addr, 
          api->index_global_addr,
          api->output_global_addr, 
          outer_num, 
          inner_num, 
          select_num,
          api->index_num, 
          0, 
          (data_type_t)api->dtype);
    } else {
      nodechip_index_select_multi_core_split_index(
          api->input_global_addr, 
          api->index_global_addr,
          api->output_global_addr, 
          outer_num, 
          inner_num, 
          select_num,
          api->index_num, 
          0, 
          (data_type_t)api->dtype);
    }
  } else {
    nodechip_index_select_multi_core (
    api->input_global_addr,
    api->index_global_addr,
    api->output_global_addr,
    input_shape,
    api->dim,
    api->index_num,
    api->axis,
    0,
    ( data_type_t ) api->dtype );
  }

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_index_select_multi_core );
#endif