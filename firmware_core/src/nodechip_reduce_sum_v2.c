#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#include "tpu_utils.h"

void nodechip_reduce_sum_2d (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int row,
int column,
int axis,
data_type_t dtype )
{
  const bool cw_trans = axis == 0;
  const int dsize = tpu_data_type_size ( dtype );
  const dim4 input_shape = { .n = 1, .c = row, .h = 1, .w = column };
  const dim4 output_shape = { .n = 1, .c = axis == 0 ? 1 : row, .h = 1, .w = axis == 1 ? 1 : column };
  const int tile = tpu_eu_num ( dtype );
  const scalar_t one_fp32 = { .f32 = 1.f };
  const scalar_t one_fp16 = tpu_fp_cast ( one_fp32, DT_FP16, DT_FP32, RM_HALF_TO_EVEN );
  const scalar_t one_fp = dtype == DT_FP16 ? one_fp16 :  one_fp32;
  const scalar_t zero = { .u32 = 0 };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  dim4 input_global_stride, output_global_stride;
  tpu_continuous_stride ( &input_global_stride, &input_shape );
  tpu_continuous_stride ( &output_global_stride, &output_shape );
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t reduce_tile_addr;
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
    int tsize = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, dtype );
    int rsize = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += rsize;
    output_local_addrs[1] = next; next += rsize;
    reduce_tile_addr = next; next += tsize;
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
      // [ 1, shape.c, DIV_UP ( shape.w, tile ), tile ] -> [ 1, shape.c, 1, tile ] -> [ 1, shape.c, 1, 1 ]
      dim2 kernel = { .h = tile_shape.h, .w = 1 };
      tpu_bdc_fp_avg_pool2d ( reduce_tile_addr, input_local_addrs[index], &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, one_fp );
      tile_shape.h = 1; kernel.h = 1; kernel.w = tile_shape.w;
      tpu_bdc_fp_avg_pool2d ( output_local_addrs[index], reduce_tile_addr, &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, one_fp );
      if ( wdone > 0 )
      {
        tpu_bdc_fp_add ( output_local_addrs[index], output_local_addrs[index], output_local_addrs[1 - index], &reduce_shape, NULL, NULL, NULL, dtype );
      }
      wtodo -= shape.w;
      wdone += shape.w;
      index = 1 - index;
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

void tpu_kernel_api_reduce_sum ( const void *args )
{
  sg_api_reduce_sum_t * api = ( sg_api_reduce_sum_t * ) args;
  data_type_t dtype = tpu_type_convert ( api->dtype );
  TPUKERNEL_ASSERT ( dtype == DT_FP32 || dtype == DT_FP16 );
  tpu_initialize();
  if ( api->reduce_dim == api->shape_dim - 1 )
  {
    int row = 1;
    for ( int i = 0; i < api->shape_dim - 1; ++i )
    {
      row *= api->shape[i];
    }
    int column = api->shape[api->shape_dim - 1];
    nodechip_reduce_sum_2d (
    api->input_global_addr,
    api->output_global_addr,
    row,
    column,
    1,
    dtype );
  }
  else if ( api->reduce_dim == 0 )
  {
    int column = 1;
    for ( int i = 1; i < api->shape_dim; ++i )
    {
      column *= api->shape[i];
    }
    int row = api->shape[0];
    nodechip_reduce_sum_2d (
    api->input_global_addr,
    api->output_global_addr,
    row,
    column,
    0,
    dtype );
  }
  else
  {
    TPUKERNEL_ASSERT ( false );
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_reduce_sum );
