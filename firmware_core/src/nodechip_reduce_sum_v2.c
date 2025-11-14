#include "sg_api_struct.h"
#include "tpu_kernel.h"


void nodechip_reduce_sum_2d (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int row,
int column,
int axis,
data_type_t dtype,
int reduction )
{
  const bool cw_trans = axis == 0;
  const int dsize = tpu_data_type_size ( dtype );
  const dim4 input_shape = { .n = 1, .c = row, .h = 1, .w = column };
  const dim4 output_shape = { .n = 1, .c = axis == 0 ? 1 : row, .h = 1, .w = axis == 1 ? 1 : column };
  const int tile = tpu_eu_num ( dtype );
  const scalar_t one_fp32 = { .f32 = 1.f };
  const scalar_t zero = { .u32 = 0 };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  dim4 input_global_stride, output_global_stride;
  tpu_continuous_stride ( &input_global_stride, &input_shape );
  tpu_continuous_stride ( &output_global_stride, &output_shape );
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t input_fp32_local_addr, reduce_tile_fp32_addrs[2];
  dim4 shape = { .n = 1, .h = 1 };
  bool l2s = false;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  dim4 l2s_shape;
  int cmax = cw_trans ? column : row;
  int wmax = cw_trans ? row : column;
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, dtype );
    int size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, DT_FP32 );
    int tsize_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    int rsize = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += rsize;
    output_local_addrs[1] = next; next += rsize;
    if ( dtype != DT_FP32 )
    {
      input_fp32_local_addr = next; next += size_fp32;
    }
    reduce_tile_fp32_addrs[0] = next; next += tsize_fp32;
    reduce_tile_fp32_addrs[1] = next; next += tsize_fp32;
    if ( ( int ) next <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( cmax > NPU_NUM )
      {
        cmax -= NPU_NUM;
        continue;
      }
      else if ( wmax > 1 )
      {
        wmax -= tile;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  int ctodo = cw_trans ? column : row;
  int cdone = 0;
  int index = 0;
  while ( ctodo != 0 )
  {
    shape.c = MIN ( cmax, ctodo );
    dim4 reduce_shape = { .n = shape.n, .c = shape.c, .h = 1, .w = 1 };
    dim4 reduce_tile_shape = { .n = shape.n, .c = shape.c, .h = 1, .w = tile };
    int wtodo = cw_trans ? row : column;
    int wdone = 0;
    while ( wtodo != 0 )
    {
      shape.w = MIN ( wmax, wtodo );
      dim4 tile_shape = { .n = shape.n, .c = shape.c, .h = DIV_UP ( shape.w, tile ), .w = tile };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      // Move input from global memory to local memory
      dim4 stride = { .n = tile_stride.n, .c = tile_stride.c, .h = 1, .w = 1 };
      if ( cw_trans )
      {
        tpu_gdma_cpy_cw_trans_S2L ( input_local_addrs[index], input_global_addr + ( cdone * input_global_stride.w + wdone * input_global_stride.c ) * dsize, &shape, &stride, &input_global_stride, dtype );
      }
      else
      {
        tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + ( cdone * input_global_stride.c + wdone * input_global_stride.w ) * dsize, &shape, &stride, &input_global_stride, dtype );
      }
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      if ( l2s )
      {
        if ( cw_trans )
        {
          tpu_gdma_cpy_cw_trans_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
        }
        else
        {
          tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
        }
        l2s = false;
      }
      // set input tail
      if ( shape.w % tile != 0 )
      {
        dim4 tail_shape = { .n = shape.n, .c = shape.c, .h = 1, .w = tile - ( shape.w % tile ) };
        tpu_bdc_set_C ( input_local_addrs[index] + shape.w * dsize, zero, &tail_shape, &tile_stride, dtype );
      }
      // [ 1, shape.c, DIV_UP ( shape.w, tile ), tile ] -> [ 1, shape.c, 1, tile ]
      dim2 kernel = { .h = tile_shape.h, .w = 1 };
      if ( dtype == DT_FP32 )
      {
        tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_addrs[index], input_local_addrs[index], &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, one_fp32 );
      }
      else
      {
        tpu_bdc_cast ( input_fp32_local_addr, input_local_addrs[index], &tile_shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_addrs[index], input_fp32_local_addr, &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, one_fp32 );
      }
      if ( wdone > 0 )
      {
        tpu_bdc_fp_add ( reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, NULL, NULL, NULL, DT_FP32 );
      }
      wtodo -= shape.w;
      wdone += shape.w;
      index = 1 - index;
    }
    scalar_t C_fp32;
    if ( reduction == 0 )
    {
      C_fp32.f32 = 1.f / ( cw_trans ? row : column );
    }
    else
    {
      C_fp32.f32 = 1.f;
    }
    if ( dtype == DT_FP32 )
    {
      dim2 kernel = { .h = 1, .w = tile };
      tpu_bdc_fp_avg_pool2d ( output_local_addrs[1 - index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, C_fp32 );
    }
    else
    {
      dim2 kernel = { .h = 1, .w = tile };
      tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, C_fp32 );
      tpu_bdc_cast ( output_local_addrs[1 - index], reduce_tile_fp32_addrs[index], &reduce_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
    }
    l2s = true;
    l2s_global_addr = output_global_addr + cdone * ( cw_trans ? output_global_stride.w : output_global_stride.c ) * dsize;
    l2s_local_addr = output_local_addrs[1 - index];
    l2s_shape.n = reduce_shape.n;
    l2s_shape.c = cw_trans ? reduce_shape.w : reduce_shape.c;
    l2s_shape.h = reduce_shape.h;
    l2s_shape.w = cw_trans ? reduce_shape.c : reduce_shape.w;
    ctodo -= shape.c;
    cdone += shape.c;
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s )
  {
    if ( cw_trans )
    {
      tpu_gdma_cpy_cw_trans_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
    }
    else
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
    }
    l2s = false;
  }
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

static inline void nodechip_reduce_sum_2d_multi_core_single_stage(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    int row,
    int column,
    int native_row,
    int native_column,
    int axis,
    data_type_t dtype,
    int reduction)
{
  const bool cw_trans = axis == 0;
  const int dsize = tpu_data_type_size ( dtype );
  const dim4 input_shape = { .n = 1, .c = row, .h = 1, .w = column };
  const dim4 output_shape = { .n = 1, .c = axis == 0 ? 1 : row, .h = 1, .w = axis == 1 ? 1 : column };
  const int tile = tpu_eu_num ( dtype );
  const scalar_t one_fp32 = { .f32 = 1.f };
  const scalar_t zero = { .u32 = 0 };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  dim4 input_global_stride, output_global_stride;
  //Notice
  if (axis == 0 ) {
    input_global_stride.w = 1;
    input_global_stride.h = input_shape.w;
    input_global_stride.c = input_shape.h * 1 * native_column;
    input_global_stride.n = input_global_stride.c * input_shape.c;
  } else
    tpu_continuous_stride ( &input_global_stride, &input_shape );
  tpu_continuous_stride ( &output_global_stride, &output_shape );
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t input_fp32_local_addr, reduce_tile_fp32_addrs[2];
  dim4 shape = { .n = 1, .h = 1 };
  bool l2s = false;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  dim4 l2s_shape;
  int cmax = cw_trans ? column : row;
  int wmax = cw_trans ? row : column;
  uint64_t next = 0;
  uint64_t size =0;
  uint64_t size_fp32 = 0;
  uint64_t tsize_fp32 = 0;
  uint64_t rsize = 0;
  while (true)
  {
    // uint64_t hw_size = DIV_UP(wmax, tile) * (uint64_t)tile;
    next = 0;
    size = (uint64_t)DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, dtype );
    size_fp32 = (uint64_t)DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, DT_FP32 );
    tsize_fp32 = (uint64_t)DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    rsize = (uint64_t)DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += rsize;
    output_local_addrs[1] = next; next += rsize;
    if ( dtype != DT_FP32 )
    {
      input_fp32_local_addr = next; next += size_fp32;
    }
    reduce_tile_fp32_addrs[0] = next; next += tsize_fp32;
    reduce_tile_fp32_addrs[1] = next; next += tsize_fp32;
    if ( next <= ( uint64_t )LOCAL_MEM_SIZE && (dtype == DT_FP32 || size_fp32 <= (uint64_t)LOCAL_MEM_SIZE))
    {
      break;
    }
    else
    {
      if ( cmax > NPU_NUM )
      {
        cmax -= NPU_NUM;
        continue;
      }
      else if ( wmax > tile )
      {
        wmax -= tile;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  int ctodo = cw_trans ? column : row;
  int cdone = 0;
  int index = 0;
  while ( ctodo != 0 )
  {
    shape.c = MIN ( cmax, ctodo );
    dim4 reduce_shape = { .n = shape.n, .c = shape.c, .h = 1, .w = 1 };
    dim4 reduce_tile_shape = { .n = shape.n, .c = shape.c, .h = 1, .w = tile };
    int wtodo = cw_trans ? row : column;
    int wdone = 0;
    while ( wtodo != 0 )
    {
      shape.w = MIN ( wmax, wtodo );
      dim4 tile_shape = { .n = shape.n, .c = shape.c, .h = DIV_UP ( shape.w, tile ), .w = tile };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      // Move input from global memory to local memory
      dim4 stride = { .n = tile_stride.n, .c = tile_stride.c, .h = 1, .w = 1 };
      if ( cw_trans )
      {
        tpu_gdma_cpy_cw_trans_S2L ( input_local_addrs[index], input_global_addr + ( cdone * input_global_stride.w + wdone * input_global_stride.c ) * dsize, &shape, &stride, &input_global_stride, dtype );
      }
      else
      {
        tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + ( cdone * input_global_stride.c + wdone * input_global_stride.w ) * dsize, &shape, &stride, &input_global_stride, dtype );
      }
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      if ( l2s )
      {
        if ( cw_trans )
        {
          tpu_gdma_cpy_cw_trans_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
        }
        else
        {
          tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
        }
        l2s = false;
      }
      // set input tail
      if ( shape.w % tile != 0 )
      {
        dim4 tail_shape = { .n = shape.n, .c = shape.c, .h = 1, .w = tile - ( shape.w % tile ) };
        tpu_bdc_set_C ( input_local_addrs[index] + shape.w * dsize, zero, &tail_shape, &tile_stride, dtype );
      }
      // [ 1, shape.c, DIV_UP ( shape.w, tile ), tile ] -> [ 1, shape.c, 1, tile ]
      dim2 kernel = { .h = tile_shape.h, .w = 1 };
      if ( dtype == DT_FP32 )
      {
        tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_addrs[index], input_local_addrs[index], &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, one_fp32 );
      }
      else if ( dtype == DT_INT8 || dtype == DT_UINT8)
      {
        tpu_bdc_int8_avg_pool2d( reduce_tile_fp32_addrs[index], input_local_addrs[index], &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, dtype, 1, 0);
      }
      else
      {
        tpu_bdc_cast ( input_fp32_local_addr, input_local_addrs[index], &tile_shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_addrs[index], input_fp32_local_addr, &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, one_fp32 );
      }
      if ( wdone > 0 )
      {
        if ( dtype == DT_INT8 || dtype == DT_UINT8 ) {
          tpu_bdc_int_add ( reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, NULL, NULL, NULL,
                            dtype, dtype, dtype, 0, RM_HALF_TO_EVEN, true );
        }
        else{
          tpu_bdc_fp_add ( reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, NULL, NULL, NULL, DT_FP32 );
        }
      }
      wtodo -= shape.w;
      wdone += shape.w;
      index = 1 - index;
    }
    scalar_t C_fp32;
    //Notice
    if ( reduction == 0 )
    {
      C_fp32.f32 = 1.f / ( cw_trans ? native_row : native_column );
    }
    else
    {
      C_fp32.f32 = 1.f;
    }
    if ( dtype == DT_FP32 )
    {
      dim2 kernel = { .h = 1, .w = tile };
      tpu_bdc_fp_avg_pool2d ( output_local_addrs[1 - index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, C_fp32 );
    }
    else if ( dtype == DT_INT8 || dtype == DT_UINT8 )
    {
      dim2 kernel = { .h = 1, .w = tile };
      tpu_bdc_int8_avg_pool2d ( output_local_addrs[1 - index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, dtype, 1, 0 );
    }
    else
    {
      dim2 kernel = { .h = 1, .w = tile };
      tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_addrs[index], reduce_tile_fp32_addrs[1 - index], &reduce_tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, C_fp32 );
      tpu_bdc_cast ( output_local_addrs[1 - index], reduce_tile_fp32_addrs[index], &reduce_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
    }
    l2s = true;
    l2s_global_addr = output_global_addr + cdone * ( cw_trans ? output_global_stride.w : output_global_stride.c ) * dsize;
    l2s_local_addr = output_local_addrs[1 - index];
    l2s_shape.n = reduce_shape.n;
    l2s_shape.c = cw_trans ? reduce_shape.w : reduce_shape.c;
    l2s_shape.h = reduce_shape.h;
    l2s_shape.w = cw_trans ? reduce_shape.c : reduce_shape.w;
    ctodo -= shape.c;
    cdone += shape.c;
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s )
  {
    if ( cw_trans )
    {
      tpu_gdma_cpy_cw_trans_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
    }
    else
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
    }
    l2s = false;
  }
}

void nodechip_reduce_sum_2d_multi_core (
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    int row,
    int column,
    int axis,
    data_type_t dtype,
    int reduction) {
  int row_real = 0, column_real = 0, native_row = row, native_column = column;
  int min_cores_needed = 1, expected_avg_slice = 0;
  int unrolling_num = axis == 1 ? native_row : native_column;
  int fixed_dim_num = axis == 1 ? native_column : native_row;

  if(axis == 1) {
    compute_current_slice_info_multi_core(unrolling_num, &row_real,
                                                         &expected_avg_slice, &min_cores_needed);
    column_real =  native_column;
  } else {
    compute_current_slice_info_multi_core(unrolling_num, &column_real,
                                                         &expected_avg_slice, &min_cores_needed);
    row_real = native_row;
  }

  const int core_idx = tpu_core_index();
  if (core_idx < min_cores_needed ) {
        int input_addr_fixed_move = axis == 0 ? 1 : fixed_dim_num;
        global_addr_t input_global_addr_new = input_global_addr + (unsigned long long)(core_idx) * input_addr_fixed_move * expected_avg_slice * tpu_data_type_size(dtype);
        global_addr_t output_global_addr_new = output_global_addr + (unsigned long long)(core_idx) * 1 * expected_avg_slice * tpu_data_type_size(dtype);
        nodechip_reduce_sum_2d_multi_core_single_stage(
          input_global_addr_new,
          output_global_addr_new,
          row_real,
          column_real,
          native_row,
          native_column,
          axis,
          dtype,
          reduction);
  }
  tpu_sync_all();
}

int tpu_kernel_api_reduce_multi_core ( const void *args )
{
  sg_api_reduce_t * api = ( sg_api_reduce_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 ||
                    api->dtype == DT_INT32 || api->dtype == DT_INT8 || api->dtype == DT_UINT8);
  TPUKERNEL_ASSERT ( api->mode == 0 || api->mode == 1 );
  tpu_initialize();
#ifdef ENABLE_MULTI_CORE
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  //Notice
  //[start_dim, end_dim)
  if ( api->end_dim == api->dim)
  {
    int row = 1;
    int column = 1;
    for ( int i = 0; i < api->start_dim; ++i )
    {
      row *= api->shape[i];
    }
    for ( int i = api->start_dim; i < api->dim; ++i )
    {
      column *= api->shape[i];
    }
    nodechip_reduce_sum_2d_multi_core (
    api->input_global_addr,
    api->output_global_addr,
    row,
    column,
    1,
    (data_type_t)api->dtype,
    api->mode );
  }
  else if ( api->start_dim == 0 )
  {
    int row = 1;
    int column = 1;
    //Notice
    for ( int i = 0; i < api->end_dim; ++i )
    {
      row *= api->shape[i];
    }
    //Notice
    for ( int i = api->end_dim; i < api->dim; ++i )
    {
      column *= api->shape[i];
    }
    nodechip_reduce_sum_2d_multi_core (
    api->input_global_addr,
    api->output_global_addr,
    row,
    column,
    0,
    (data_type_t)api->dtype,
    api->mode );
  }
  else
  {
    TPUKERNEL_ASSERT ( false );
  }
  tpu_poll();
  return 0;

#else
  if ( api->end_dim == api->dim )
  {
    int row = 1;
    int column = 1;
    for ( int i = 0; i < api->start_dim; ++i )
    {
      row *= api->shape[i];
    }
    for ( int i = api->start_dim; i < api->dim; ++i )
    {
      column *= api->shape[i];
    }
    nodechip_reduce_sum_2d (
    api->input_global_addr,
    api->output_global_addr,
    row,
    column,
    1,
    ( data_type_t ) api->dtype,
    api->mode );
  }
  else if ( api->start_dim == 0 )
  {
    int row = 1;
    int column = 1;
    for ( int i = 0; i < api->end_dim; ++i )
    {
      row *= api->shape[i];
    }
    for ( int i = api->end_dim; i < api->dim; ++i )
    {
      column *= api->shape[i];
    }
    nodechip_reduce_sum_2d (
    api->input_global_addr,
    api->output_global_addr,
    row,
    column,
    0,
    ( data_type_t ) api->dtype,
    api->mode );
  }
  else
  {
    TPUKERNEL_ASSERT ( false );
  }
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_reduce_multi_core );