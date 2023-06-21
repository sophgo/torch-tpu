#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

void nodechip_softmax_forward_2DR1_no_max_pivot_parallel (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int batch_num, // row number
int class_num, // column number
data_type_t dtype ) // DT_FP32 or DT_FP16
{
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  const dim4 input_shape = { .n = 1, .c = batch_num, .h = 1, .w = class_num };
  const dim4 output_shape = { .n = 1, .c = batch_num, .h = 1, .w = class_num };
  int cmax = batch_num, wmax = class_num;
  dim4 input_global_stride, output_global_stride;
  tpu_continuous_stride ( &input_global_stride, &input_shape );
  tpu_continuous_stride ( &output_global_stride, &output_shape );
  local_addr_t next_base = 0;
  local_addr_t exp_coeff_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t exp_table_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t input_exp_local_addr;
  local_addr_t reduce_tile_local_addr, reduce_sum_local_addrs[2];
  local_addr_t work0_local_addr, work1_local_addr;
  local_addr_t next;
  while ( true )
  {
    int size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, dtype );
    int size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, DT_FP32 );
    int reduce_tile_size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    int reduce_size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    next = next_base;
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    input_exp_local_addr = next; next += size_fp32;
    reduce_tile_local_addr = next; next += reduce_tile_size_fp32;
    reduce_sum_local_addrs[0] = next; next += reduce_size_fp32;
    reduce_sum_local_addrs[1] = next; next += reduce_size_fp32;
    work0_local_addr = next; next += size_fp32;
    work1_local_addr = next; next += size_fp32;
    output_local_addrs[0] = next; next += size;
    output_local_addrs[1] = next; next += size;
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
  tpu_bdc_load_fp32_exp_coeff ( exp_coeff_addr );
  tpu_bdc_load_fp32_exp_table ( exp_table_addr );
  bool l2s = false;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  dim4 l2s_shape, l2s_global_stride, l2s_local_stride;
  dim4 shape = { .n = 1, .h = 1 };
  // set output zeros
  int index = 0; // Ping-pong buffer switching index
  int ctodo = batch_num, cdone = 0;
  while ( ctodo != 0 )
  {
    shape.c = MIN ( cmax, ctodo );
    dim4 reduce_shape = { .n = 1, .c = shape.c, .h = 1, .w = 1 };
    int wtodo = class_num, wdone = 0;
    while ( wtodo != 0 )
    {
      shape.w = MIN ( wmax, wtodo );
      dim4 tile_shape = { .n = 1, .c = shape.c, .h = DIV_UP ( shape.w, tile ), .w = tile };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      // Move input from global memory to local memory
      tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + ( 1UL * cdone * input_global_stride.c + 1UL * wdone * input_global_stride.w ) * dsize, &shape, &tile_stride, &input_global_stride, dtype );
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      if ( l2s )
      {
        tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &l2s_global_stride, &l2s_local_stride, dtype );
        l2s = false;
      }
      // input FP16 -> FP32
      if ( dtype == DT_FP16 )
      {
        tpu_bdc_cast ( input_exp_local_addr, input_local_addrs[index], &tile_shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        // input = exp ( input )
        tpu_bdc_fp32_exp ( input_exp_local_addr, input_exp_local_addr, work0_local_addr, work1_local_addr, exp_coeff_addr, exp_table_addr, &tile_shape );
      }
      else
      {
        // input = exp ( input )
        tpu_bdc_fp32_exp ( input_exp_local_addr, input_local_addrs[index], work0_local_addr, work1_local_addr, exp_coeff_addr, exp_table_addr, &tile_shape );
      }
      // set input tail
      if ( shape.w % tile != 0 )
      {
        dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, DT_FP32 );
        dim4 tail_shape = { .n = 1, .c = shape.c, .h = 1, .w = tile - ( shape.w % tile ) };
        tpu_bdc_set_C ( input_exp_local_addr + shape.w * sizeof ( float ), zero, &tail_shape, &tile_stride, DT_FP32 );
      }
      // input_sum = sum ( input )
      // [ 1, shape.c, DIV_UP ( shape.w, tile ), tile ] -> [ 1, shape.c, 1, tile ] -> [ 1, shape.c, 1, 1 ]
      dim2 kernel = { .h = tile_shape.h, .w = 1 };
      tpu_bdc_fp_avg_pool2d ( reduce_tile_local_addr, input_exp_local_addr, &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, one_fp32 );
      tile_shape.h = 1; kernel.h = 1; kernel.w = tile_shape.w;
      tpu_bdc_fp_avg_pool2d ( reduce_sum_local_addrs[index], reduce_tile_local_addr, &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, DT_FP32, one_fp32 );
      if ( wdone > 0 )
      {
        tpu_bdc_fp_add ( reduce_sum_local_addrs[index], reduce_sum_local_addrs[index], reduce_sum_local_addrs[1 - index], &reduce_shape, NULL, NULL, NULL, DT_FP32 );
      }
      wtodo -= shape.w;
      wdone += shape.w;
      index = 1 - index;
    }
    local_addr_t reduce_sum_local_addr = reduce_sum_local_addrs[1 - index];
    // input_sum = 1 / input_sum
    tpu_bdc_fp32_reciprocal ( reduce_sum_local_addr, reduce_sum_local_addr, &reduce_shape, NULL, NULL );
    wtodo = class_num, wdone = 0;
    while ( wtodo != 0 )
    {
      shape.w = MIN ( wmax, wtodo );
      dim4 tile_shape = { .n = 1, .c = shape.c, .h = DIV_UP ( shape.w, tile ), .w = tile };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      if ( wmax != class_num )
      {
        // Move input from global memory to local memory
        tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + ( 1UL * cdone * input_global_stride.c + 1UL * wdone * input_global_stride.w ) * dsize, &shape, &tile_stride, &input_global_stride, dtype );
      }
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      if ( l2s )
      {
        tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &l2s_global_stride, &l2s_local_stride, dtype );
        l2s = false;
      }
      if ( wmax != class_num )
      {
        // input FP16 -> FP32
        if ( dtype == DT_FP16 )
        {
          tpu_bdc_cast ( input_exp_local_addr, input_local_addrs[index], &tile_shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
          // input = exp ( input )
          tpu_bdc_fp32_exp ( input_exp_local_addr, input_exp_local_addr, work0_local_addr, work1_local_addr, exp_coeff_addr, exp_table_addr, &tile_shape );
        }
        else
        {
          // input = exp ( input )
          tpu_bdc_fp32_exp ( input_exp_local_addr, input_local_addrs[index], work0_local_addr, work1_local_addr, exp_coeff_addr, exp_table_addr, &tile_shape );
        }
      }
      // output = input * input_sum
      dim4 reduce_stride; tpu_aligned_stride ( &reduce_stride, 0, &reduce_shape, DT_FP32 );
      dim4 reduce_bcast_stride = { .n = reduce_stride.n, .c = reduce_stride.c, 0, 0 };
      if ( dtype == DT_FP16 )
      {
        tpu_bdc_fp_mul ( input_exp_local_addr, input_exp_local_addr, reduce_sum_local_addr, &tile_shape, NULL, NULL, &reduce_bcast_stride, DT_FP32 );
        tpu_bdc_cast ( output_local_addrs[index], input_exp_local_addr, &tile_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
      }
      else
      {
        tpu_bdc_fp_mul ( output_local_addrs[index], input_exp_local_addr, reduce_sum_local_addr, &tile_shape, NULL, NULL, &reduce_bcast_stride, dtype );
      }
      // Move output from local memory to global memory
      dim4 stride = { .n = tile_stride.n, .c = tile_stride.c, .h = 1, .w = 1 };
      l2s = true;
      l2s_global_addr = output_global_addr + ( cdone * output_global_stride.c + wdone * output_global_stride.w ) * dsize;
      l2s_local_addr = output_local_addrs[index];
      l2s_shape = shape;
      l2s_global_stride = output_global_stride;
      l2s_local_stride = stride;
      wtodo -= shape.w;
      wdone += shape.w;
      index = 1 - index;
    }
    ctodo -= shape.c;
    cdone += shape.c;
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s )
  {
    tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &l2s_global_stride, &l2s_local_stride, dtype );
    l2s = false;
  }
}
