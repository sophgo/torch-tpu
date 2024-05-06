#include "sg_api_struct.h"
#include "tpu_kernel.h"


#define DEFAULT_LOCAL_ADDR 0xFFFFFFFF

static inline bool nodechip_softmax_2dr1_max_pivot_split_row_only (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int row,
int column,
data_type_t dtype )
{
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  scalar_t neg_inf;
  if ( dtype == DT_FP32 )
  {
    neg_inf.u32 = 0xff800000; // 1111 1111 1000 0000 0000 0000 0000 0000
  }
  else if ( dtype == DT_FP16 )
  {
    neg_inf.u32 = 0x0000fc00; // 0000 0000 0000 0000 1111 1100 0000 0000
  }
  else if ( dtype == DT_BFP16 )
  {
    neg_inf.u32 = 0x0000ff80; // 0000 0000 0000 0000 1111 1111 1000 0000
  }
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  const dim4 shape = { .n = 1, .c = row, .h = 1, .w = column };
  dim4 input_global_stride; tpu_continuous_stride ( &input_global_stride, &shape );
  dim4 output_global_stride; tpu_continuous_stride ( &output_global_stride, &shape );
  local_addr_t input_local_addrs[2] = { DEFAULT_LOCAL_ADDR, DEFAULT_LOCAL_ADDR };
  local_addr_t output_local_addrs[2] = { DEFAULT_LOCAL_ADDR, DEFAULT_LOCAL_ADDR };
  local_addr_t input_max_tile_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t input_max_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t input_sum_tile_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t input_sum_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t work0_local_addr = DEFAULT_LOCAL_ADDR, work1_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t exp_coeff_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t input_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t next = 0;
  int cmax = row;
  while ( true )
  {
    const int input_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( column, tile ), tile, dtype );
    const int input_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( column, tile ), tile, DT_FP32 );
    const int input_tile_max_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, dtype );
    const int input_max_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    const int input_sum_tile_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    const int input_sum_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    const int exp_coeff_size = tpu_aligned_feature_size ( 1, 10, dtype );
    next = 0;
    exp_coeff_local_addr = next; next += exp_coeff_size;
    input_local_addrs[0] = next; next += input_size;
    input_local_addrs[1] = next; next += input_size;
    input_max_tile_local_addr = next; next += input_tile_max_size;
    input_max_local_addr = next; next += input_max_size;
    if ( dtype != DT_FP32 )
    {
      input_fp32_local_addr = next; next += input_fp32_size;
    }
    input_sum_tile_fp32_local_addr = next; next += input_sum_tile_fp32_size;
    input_sum_fp32_local_addr = next; next += input_sum_fp32_size;
    work0_local_addr = next; next += input_size;
    work1_local_addr = next; next += input_size;
    output_local_addrs[0] = next; next += input_size;
    output_local_addrs[1] = next; next += input_size;
    if ( ( int ) next > LOCAL_MEM_SIZE )
    {
      if ( cmax > NPU_NUM )
      {
        cmax -= ( ( cmax % NPU_NUM > 0 ) ? ( cmax % NPU_NUM ) : NPU_NUM );
        continue;
      }
      else
      {
        return false;
      }
    }
    else
    {
      break;
    }
  }
  tpu_bdc_load_fp_exp_coeff ( exp_coeff_local_addr, dtype );
  bool l2s = false;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = DEFAULT_LOCAL_ADDR;
  dim4 l2s_shape;
  dim4 work_shape = { .n = 1, .h = 1, .w = column };
  int index = 0;
  int ctodo = row, cdone = 0;
  while ( ctodo != 0 )
  {
    work_shape.c = MIN ( ctodo, cmax );
    const dim4 tile_shape =
    {
      .n = 1,
      .c = work_shape.c,
      .h = DIV_UP ( column, tile ),
      .w = tile
    };
    const dim4 reduce_tile_shape =
    {
      .n = 1,
      .c = work_shape.c,
      .h = 1,
      .w = tile
    };
    const dim4 reduce_shape =
    {
      .n = 1,
      .c = work_shape.c,
      .h = 1,
      .w = 1
    };
    const dim4 tail_shape =
    {
      .n = 1,
      .c = work_shape.c,
      .h = 1,
      .w = tile - ( column % tile )
    };
    dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
    dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
    dim4 reduce_stride; tpu_aligned_stride ( &reduce_stride, 0, &reduce_shape, dtype );
    dim4 reduce_bcast_stride =
    {
      .n = reduce_stride.n,
      .c = reduce_stride.c,
      .h = 0,
      .w = 0
    };
    dim4 reduce_fp32_stride; tpu_aligned_stride ( &reduce_fp32_stride, 0, &reduce_shape, DT_FP32 );
    dim4 reduce_bcast_fp32_stride =
    {
      .n = reduce_fp32_stride.n,
      .c = reduce_fp32_stride.c,
      .h = 0,
      .w = 0
    };
    const dim2 reduce_tile_kernel =
    {
      .h = tile_shape.h,
      .w = 1
    };
    const dim2 reduce_kernel =
    {
      .h = 1,
      .w = tile
    };
    // Move Input from global memory to local memory
    tpu_gdma_cpy_S2L ( input_local_addrs[index],
                       input_global_addr + 1UL * cdone * input_global_stride.c * dsize,
                       &work_shape,
                       NULL,
                       &input_global_stride,
                       dtype );
    // Synchronize Point
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    // Move Output from local memory to global memory
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr,
                         l2s_local_addr,
                         &l2s_shape,
                         &output_global_stride,
                         NULL,
                         dtype );
      l2s = false;
    }
    // Set tail -inf
    if ( column % tile != 0 )
    {
      tpu_bdc_set_C ( input_local_addrs[index] + column * dsize,
                      neg_inf,
                      &tail_shape,
                      &tile_stride,
                      dtype );
    }
    // [ 1, work_shape.c, DIV_UP ( column, tile ), tile ] -> [ 1, work_shape.c, 1, tile ]
    tpu_bdc_fp_max_pool2d ( input_max_tile_local_addr,
                            input_local_addrs[index],
                            &tile_shape,
                            &reduce_tile_kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            dtype,
                            neg_inf );
    // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
    tpu_bdc_fp_max_pool2d ( input_max_local_addr,
                            input_max_tile_local_addr,
                            &reduce_tile_shape,
                            &reduce_kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            dtype,
                            neg_inf );
    // Input = Input - Input_Max
    tpu_bdc_fp_sub ( input_local_addrs[index],
                     input_local_addrs[index],
                     input_max_local_addr,
                     &tile_shape,
                     NULL,
                     NULL,
                     &reduce_bcast_stride,
                     dtype );
    // Input = EXP ( Input )
    tpu_bdc_fp_exp ( input_local_addrs[index],
                     input_local_addrs[index],
                     work0_local_addr,
                     work1_local_addr,
                     exp_coeff_local_addr,
                     &tile_shape,
                     dtype );
    // Input_FP32 = FP32 ( Input )
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast ( input_fp32_local_addr,
                     input_local_addrs[index],
                     &tile_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
    }
    else
    {
      input_fp32_local_addr = input_local_addrs[index];
    }
    // Set tail zero
    if ( column % tile != 0 )
    {
      tpu_bdc_set_C ( input_fp32_local_addr + column * sizeof ( float ),
                      zero,
                      &tail_shape,
                      &tile_fp32_stride,
                      DT_FP32 );
    }
    // [ 1, work_shape.c, DIV_UP ( column, tile ), tile ] -> [ 1, work_shape.c, 1, tile ]
    tpu_bdc_fp_avg_pool2d ( input_sum_tile_fp32_local_addr,
                            input_fp32_local_addr,
                            &tile_shape,
                            &reduce_tile_kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            DT_FP32,
                            one_fp32 );
    // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
    tpu_bdc_fp_avg_pool2d ( input_sum_fp32_local_addr,
                            input_sum_tile_fp32_local_addr,
                            &reduce_tile_shape,
                            &reduce_kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            DT_FP32,
                            one_fp32 );
    // Input_FP32 = Input_FP32 / Input_Sum_FP32
    tpu_bdc_fp32_reciprocal ( input_sum_fp32_local_addr,
                              input_sum_fp32_local_addr,
                              &reduce_shape,
                              NULL,
                              NULL );
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_fp_mul ( input_fp32_local_addr,
                       input_fp32_local_addr,
                       input_sum_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &reduce_bcast_fp32_stride,
                       DT_FP32 );
      tpu_bdc_cast ( output_local_addrs[index],
                     input_fp32_local_addr,
                     &tile_shape,
                     NULL,
                     NULL,
                     dtype,
                     DT_FP32,
                     RM_HALF_TO_EVEN );
    }
    else
    {
      tpu_bdc_fp_mul ( output_local_addrs[index],
                       input_fp32_local_addr,
                       input_sum_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &reduce_bcast_fp32_stride,
                       DT_FP32 );
    }
    l2s = true;
    l2s_local_addr = output_local_addrs[index];
    l2s_global_addr = output_global_addr + 1UL * cdone * output_global_stride.c * dsize;
    l2s_shape = work_shape;
    index = 1 - index;
    ctodo -= work_shape.c;
    cdone += work_shape.c;
  }
  // Synchronize Point
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  // Move Output from local memory to global memory
  if ( l2s )
  {
    tpu_gdma_cpy_L2S ( l2s_global_addr,
                       l2s_local_addr,
                       &l2s_shape,
                       &output_global_stride,
                       NULL,
                       dtype );
    l2s = false;
  }
  return true;
}

static inline bool nodechip_softmax_2dr1_max_pivot (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int row,
int column,
data_type_t dtype )
{
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  scalar_t neg_inf;
  if ( dtype == DT_FP32 )
  {
    neg_inf.u32 = 0xff800000; // 1111 1111 1000 0000 0000 0000 0000 0000
  }
  else if ( dtype == DT_FP16 )
  {
    neg_inf.u32 = 0x0000fc00; // 0000 0000 0000 0000 1111 1100 0000 0000
  }
  else if ( dtype == DT_BFP16 )
  {
    neg_inf.u32 = 0x0000ff80; // 0000 0000 0000 0000 1111 1111 1000 0000
  }
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  const dim4 shape = { .n = 1, .c = row, .h = 1, .w = column };
  dim4 input_global_stride; tpu_continuous_stride ( &input_global_stride, &shape );
  dim4 output_global_stride; tpu_continuous_stride ( &output_global_stride, &shape );
  local_addr_t input_local_addrs[2] = { DEFAULT_LOCAL_ADDR, DEFAULT_LOCAL_ADDR };
  local_addr_t output_local_addrs[2] = { DEFAULT_LOCAL_ADDR, DEFAULT_LOCAL_ADDR };
  local_addr_t input_max_tile_local_addrs[2] = { DEFAULT_LOCAL_ADDR, DEFAULT_LOCAL_ADDR };
  local_addr_t input_max_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t input_sum_tile_fp32_local_addrs[2] = { DEFAULT_LOCAL_ADDR, DEFAULT_LOCAL_ADDR };
  local_addr_t input_sum_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t work0_local_addr = DEFAULT_LOCAL_ADDR, work1_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t exp_coeff_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t input_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t next = 0;
  int wmax = column;
  while ( true )
  {
    const int input_size = tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, dtype );
    const int input_fp32_size = tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, DT_FP32 );
    const int input_tile_max_size = tpu_aligned_feature_size ( 1, tile, dtype );
    const int input_max_size = tpu_aligned_feature_size ( 1, 1, dtype );
    const int input_sum_tile_fp32_size = tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    const int input_sum_fp32_size = tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    const int exp_coeff_size = tpu_aligned_feature_size ( 1, 10, dtype );
    next = 0;
    exp_coeff_local_addr = next; next += exp_coeff_size;
    input_local_addrs[0] = next; next += input_size;
    input_local_addrs[1] = next; next += input_size;
    input_max_tile_local_addrs[0] = next; next += input_tile_max_size;
    input_max_tile_local_addrs[1] = next; next += input_tile_max_size;
    input_max_local_addr = next; next += input_max_size;
    if ( dtype != DT_FP32 )
    {
      input_fp32_local_addr = next; next += input_fp32_size;
    }
    input_sum_tile_fp32_local_addrs[0] = next; next += input_sum_tile_fp32_size;
    input_sum_tile_fp32_local_addrs[1] = next; next += input_sum_tile_fp32_size;
    input_sum_fp32_local_addr = next; next += input_sum_fp32_size;
    work0_local_addr = next; next += input_size;
    work1_local_addr = next; next += input_size;
    output_local_addrs[0] = next; next += input_size;
    output_local_addrs[1] = next; next += input_size;
    if ( ( int ) next > LOCAL_MEM_SIZE )
    {
      if ( wmax > 1 )
      {
        wmax /= 2;
        continue;
      }
      else
      {
        return false;
      }
    }
    else
    {
      break;
    }
  }
  tpu_bdc_load_fp_exp_coeff ( exp_coeff_local_addr, dtype );
  bool l2s = false;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = DEFAULT_LOCAL_ADDR;
  dim4 l2s_shape;
  dim4 work_shape = { .n = 1, .h = 1 };
  int index = 0;
  int ctodo = row, cdone = 0;
  while ( ctodo != 0 )
  {
    work_shape.c = MIN ( ctodo, NPU_NUM );
    const dim4 reduce_tile_shape =
    {
      .n = 1,
      .c = work_shape.c,
      .h = 1,
      .w = tile
    };
    const dim4 reduce_shape =
    {
      .n = 1,
      .c = work_shape.c,
      .h = 1,
      .w = 1
    };
    dim4 reduce_stride; tpu_aligned_stride ( &reduce_stride, 0, &reduce_shape, dtype );
    dim4 reduce_bcast_stride =
    {
      .n = reduce_stride.n,
      .c = reduce_stride.c,
      .h = 0,
      .w = 0
    };
    dim4 reduce_fp32_stride; tpu_aligned_stride ( &reduce_fp32_stride, 0, &reduce_shape, DT_FP32 );
    dim4 reduce_bcast_fp32_stride =
    {
      .n = reduce_fp32_stride.n,
      .c = reduce_fp32_stride.c,
      .h = 0,
      .w = 0
    };
    const dim2 reduce_kernel =
    {
      .h = 1,
      .w = tile
    };
    int wtodo = column, wdone = 0;
    while ( wtodo != 0 )
    {
      work_shape.w = MIN ( wtodo, wmax );
      const dim4 tile_shape =
      {
        .n = 1,
        .c = work_shape.c,
        .h = DIV_UP ( work_shape.w, tile ),
        .w = tile
      };
      const dim4 tail_shape =
      {
        .n = 1,
        .c = work_shape.c,
        .h = 1,
        .w = tile - ( work_shape.w % tile )
      };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
      const dim2 reduce_tile_kernel =
      {
        .h = tile_shape.h,
        .w = 1
      };
      // Move Input from global memory to local memory
      tpu_gdma_cpy_S2L ( input_local_addrs[index],
                         input_global_addr + ( 1UL * cdone * input_global_stride.c + 1UL * wdone * input_global_stride.w ) * dsize,
                         &work_shape,
                         NULL,
                         &input_global_stride,
                         dtype );
      // Synchronize Point
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      // Set tail -inf
      if ( work_shape.w % tile != 0 )
      {
        tpu_bdc_set_C ( input_local_addrs[index] + work_shape.w * dsize,
                        neg_inf,
                        &tail_shape,
                        &tile_stride,
                        dtype );
      }
      // [ 1, work_shape.c, DIV_UP ( work_shape.w, tile ), tile ] -> [ 1, work_shape.c, 1, tile ]
      tpu_bdc_fp_max_pool2d ( input_max_tile_local_addrs[wdone > 0 ? 0 : 1],
                              input_local_addrs[index],
                              &tile_shape,
                              &reduce_tile_kernel,
                              &zero_pad,
                              &stride_one,
                              &dilation_one,
                              dtype,
                              neg_inf );
      if ( wdone > 0 )
      {
        tpu_bdc_max ( input_max_tile_local_addrs[1],
                      input_max_tile_local_addrs[1],
                      input_max_tile_local_addrs[0],
                      &reduce_tile_shape,
                      NULL,
                      NULL,
                      NULL,
                      dtype );
      }
      index = 1 - index;
      wtodo -= work_shape.w;
      wdone += work_shape.w;
    }
    // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
    tpu_bdc_fp_max_pool2d ( input_max_local_addr,
                            input_max_tile_local_addrs[1],
                            &reduce_tile_shape,
                            &reduce_kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            dtype,
                            neg_inf );
    wtodo = column, wdone = 0;
    while ( wtodo != 0 )
    {
      work_shape.w = MIN ( wtodo, wmax );
      const dim4 tile_shape =
      {
        .n = 1,
        .c = work_shape.c,
        .h = DIV_UP ( work_shape.w, tile ),
        .w = tile
      };
      const dim4 tail_shape =
      {
        .n = 1,
        .c = work_shape.c,
        .h = 1,
        .w = tile - ( work_shape.w % tile )
      };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
      const dim2 reduce_tile_kernel =
      {
        .h = tile_shape.h,
        .w = 1
      };
      // Move Input from global memory to local memory
      tpu_gdma_cpy_S2L ( input_local_addrs[index],
                         input_global_addr + ( 1UL * cdone * input_global_stride.c + 1UL * wdone * input_global_stride.w ) * dsize,
                         &work_shape,
                         NULL,
                         &input_global_stride,
                         dtype );
      // Synchronize Point
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      // Input = Input - Input_Max
      tpu_bdc_fp_sub ( input_local_addrs[index],
                       input_local_addrs[index],
                       input_max_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &reduce_bcast_stride,
                       dtype );
      // Input = EXP ( Input )
      tpu_bdc_fp_exp ( input_local_addrs[index],
                       input_local_addrs[index],
                       work0_local_addr,
                       work1_local_addr,
                       exp_coeff_local_addr,
                       &tile_shape,
                       dtype );
      // Input_FP32 = FP32 ( Input )
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_cast ( input_fp32_local_addr,
                       input_local_addrs[index],
                       &tile_shape,
                       NULL,
                       NULL,
                       DT_FP32,
                       dtype,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        input_fp32_local_addr = input_local_addrs[index];
      }
      // Set tail zero
      if ( work_shape.w % tile != 0 )
      {
        tpu_bdc_set_C ( input_fp32_local_addr + work_shape.w * sizeof ( float ),
                        zero,
                        &tail_shape,
                        &tile_fp32_stride,
                        DT_FP32 );
      }
      // [ 1, work_shape.c, DIV_UP ( work_shape.w, tile ), tile ] -> [ 1, work_shape.c, 1, tile ]
      tpu_bdc_fp_avg_pool2d ( input_sum_tile_fp32_local_addrs[wdone > 0 ? 0 : 1],
                              input_fp32_local_addr,
                              &tile_shape,
                              &reduce_tile_kernel,
                              &zero_pad,
                              &stride_one,
                              &dilation_one,
                              DT_FP32,
                              one_fp32 );
      if ( wdone > 0 )
      {
        tpu_bdc_fp_add ( input_sum_tile_fp32_local_addrs[1],
                         input_sum_tile_fp32_local_addrs[1],
                         input_sum_tile_fp32_local_addrs[0],
                         &reduce_tile_shape,
                         NULL,
                         NULL,
                         NULL,
                         DT_FP32 );
      }
      index = 1 - index;
      wtodo -= work_shape.w;
      wdone += work_shape.w;
    }
    // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
    tpu_bdc_fp_avg_pool2d ( input_sum_fp32_local_addr,
                            input_sum_tile_fp32_local_addrs[1],
                            &reduce_tile_shape,
                            &reduce_kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            DT_FP32,
                            one_fp32 );
    // Input_FP32 = Input_FP32 / Input_Sum_FP32
    tpu_bdc_fp32_reciprocal ( input_sum_fp32_local_addr,
                              input_sum_fp32_local_addr,
                              &reduce_shape,
                              NULL,
                              NULL );
    wtodo = column, wdone = 0;
    while ( wtodo != 0 )
    {
      work_shape.w = MIN ( wtodo, wmax );
      const dim4 tile_shape =
      {
        .n = 1,
        .c = work_shape.c,
        .h = DIV_UP ( work_shape.w, tile ),
        .w = tile
      };
      dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
      dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
      // Move Input from global memory to local memory
      tpu_gdma_cpy_S2L ( input_local_addrs[index],
                         input_global_addr + ( 1UL * cdone * input_global_stride.c + 1UL * wdone * input_global_stride.w ) * dsize,
                         &work_shape,
                         NULL,
                         &input_global_stride,
                         dtype );
      // Synchronize Point
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      tpu_parallel_start();
      // Move Output from local memory to global memory
      if ( l2s )
      {
        tpu_gdma_cpy_L2S ( l2s_global_addr,
                           l2s_local_addr,
                           &l2s_shape,
                           &output_global_stride,
                           NULL,
                           dtype );
        l2s = false;
      }
      // Input = Input - Input_Max
      tpu_bdc_fp_sub ( input_local_addrs[index],
                       input_local_addrs[index],
                       input_max_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &reduce_bcast_stride,
                       dtype );
      // Input = EXP ( Input )
      tpu_bdc_fp_exp ( input_local_addrs[index],
                       input_local_addrs[index],
                       work0_local_addr,
                       work1_local_addr,
                       exp_coeff_local_addr,
                       &tile_shape,
                       dtype );
      // Input_FP32 = FP32 ( Input )
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_cast ( input_fp32_local_addr,
                       input_local_addrs[index],
                       &tile_shape,
                       NULL,
                       NULL,
                       DT_FP32,
                       dtype,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        input_fp32_local_addr = input_local_addrs[index];
      }
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_fp_mul ( input_fp32_local_addr,
                         input_fp32_local_addr,
                         input_sum_fp32_local_addr,
                         &tile_shape,
                         NULL,
                         NULL,
                         &reduce_bcast_fp32_stride,
                         DT_FP32 );
        tpu_bdc_cast ( output_local_addrs[index],
                       input_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       dtype,
                       DT_FP32,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        tpu_bdc_fp_mul ( output_local_addrs[index],
                         input_fp32_local_addr,
                         input_sum_fp32_local_addr,
                         &tile_shape,
                         NULL,
                         NULL,
                         &reduce_bcast_fp32_stride,
                         DT_FP32 );
      }
      l2s = true;
      l2s_local_addr = output_local_addrs[index];
      l2s_global_addr = output_global_addr + ( 1UL * cdone * output_global_stride.c + 1UL * wdone * output_global_stride.w ) * dsize;
      l2s_shape = work_shape;
      index = 1 - index;
      wtodo -= work_shape.w;
      wdone += work_shape.w;
    }
    ctodo -= work_shape.c;
    cdone += work_shape.c;
  }
  // Synchronize Point
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  // Move Output from local memory to global memory
  if ( l2s )
  {
    tpu_gdma_cpy_L2S ( l2s_global_addr,
                       l2s_local_addr,
                       &l2s_shape,
                       &output_global_stride,
                       NULL,
                       dtype );
    l2s = false;
  }
  return true;
}


extern void nodechip_softmax ( global_addr_t bottom_global_offset,
                               global_addr_t top_global_offset,
                               const int * shape,
                               int dims,
                               int beg_axis,
                               int end_axis,
                               int log,
                               float scale_val,
                               data_type_t dtype );

void nodechip_softmax_forward_single_core_v2 (
  global_addr_t input_global_addr,
  global_addr_t output_global_addr,
  int*          shape,
  int           dim,
  int           axis,
  data_type_t   dtype ) {
  if ( axis == dim - 1 )
  {
    int row = 1;
    for ( int i = 0; i < dim - 1; ++i )
    {
      row *= shape[i];
    }
    int column = shape[dim - 1];
    bool ret = nodechip_softmax_2dr1_max_pivot_split_row_only (
               input_global_addr,
               output_global_addr,
               row,
               column,
               ( data_type_t ) dtype );
    if ( !ret )
    {
      ret = nodechip_softmax_2dr1_max_pivot (
            input_global_addr,
            output_global_addr,
            row,
            column,
            ( data_type_t ) dtype );
    }
    TPUKERNEL_ASSERT ( ret );
  }
  else
  {
    nodechip_softmax ( input_global_addr,
                       output_global_addr,
                       shape,
                       dim,
                       axis,
                       axis,
                       0,
                       1.f,
                       ( data_type_t ) dtype );
  }
}

void tpu_kernel_api_softmax ( const void * args )
{
  sg_api_softmax_t * api = ( sg_api_softmax_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_softmax_forward_single_core_v2(
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    api->axis,
    api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax );


#ifdef BACKEND_SG2260
extern void nodechip_softmax_forward_multi_core (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int*          shape,
int           dims,
int           begin_dim,
int           end_dim,
float         scale_val,
data_type_t   dtype );

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

void nodechip_softmax_forward_multi_core_v2 (
  global_addr_t input_global_addr,
  global_addr_t output_global_addr,
  int*          shape,
  int           dim,
  int           axis,
  data_type_t   dtype ) {
    dim4 new_shape = {.n = 1, .c =1, .h= 1, .w=1};
    // new_shape.h = shape[axis];
    for (int i=0; i < axis; i++)
      new_shape.c*=shape[i];
    for (int i=axis; i < dim; i++)
      new_shape.w*=shape[i];
    int outer_num =  new_shape.c;
    int outer_num_real = 1, outer_num_avg =1;
    int min_cores_needed = 1;
    compute_current_slice_info_multi_core(outer_num, &outer_num_real, &outer_num_avg, &min_cores_needed);
    const int core_idx = tpu_core_index();
    if(core_idx < min_cores_needed) {
          //Shape:  [n',c',h',w'] ->  [1. slice_intracores, 1, size_from_axis]
          //Input:  [1, c, h, w]  ->  [1, c_sliced, 1, size_from_axis]
          //output: [1, c, h, w]  ->  [1, c_sliced, 1, size_from_axis]
          new_shape.c = outer_num_real;
          int  sliced_shape[4];
          sliced_shape[0]=new_shape.n;
          sliced_shape[1]=new_shape.c;
          sliced_shape[2]=new_shape.h;
          sliced_shape[3]=new_shape.w;
          nodechip_softmax_forward_single_core_v2(
            input_global_addr + core_idx * outer_num_avg * new_shape.h * new_shape.w * tpu_data_type_size(dtype),
            output_global_addr + core_idx * outer_num_avg *  new_shape.h  * new_shape.w  * tpu_data_type_size(dtype),
            sliced_shape,
            4,
            3,
            dtype);
    }
}

void tpu_kernel_api_softmax_multi_core ( const void *args )
{
  sg_api_softmax_t *api = ( sg_api_softmax_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_softmax_forward_multi_core_v2 (
   api->input_global_addr,
   api->output_global_addr,
   api->shape,
   api->dim,
   api->axis,
   ( data_type_t ) api->dtype );
  //  nodechip_softmax_forward_multi_core (
  //     api->input_global_addr,
  //     api->output_global_addr,
  //     api->shape,
  //     api->dim,
  //     api->axis,
  //     api->axis,
  //     1.f,
  //     ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax_multi_core );
#endif

void tpu_kernel_api_log_softmax(const void *args) {
  sg_api_log_softmax_t *api = (sg_api_log_softmax_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  if (api->axis != api->dim - 1) {
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }
  tpu_initialize();
  nodechip_softmax(api->input_global_addr, api->output_global_addr, api->shape,
                   api->dim, api->axis, api->axis, 1, 1.f,
                   (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_log_softmax);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_log_softmax_multi_core(const void *args) {
  sg_api_log_softmax_t *api = (sg_api_log_softmax_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  if (api->axis != api->dim - 1) {
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }

  dim4 new_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
  // new_shape.h = shape[axis];
  for (int i = 0; i < api->axis; i++) {
    new_shape.c *= api->shape[i];
  }

  for (int i = api->axis; i < api->dim; i++) {
    new_shape.w *= api->shape[i];
  }

  int outer_num = new_shape.c;
  int outer_num_real = 1, outer_num_avg = 1;
  int min_cores_needed = 1;
  compute_current_slice_info_multi_core(outer_num, &outer_num_real,
                                        &outer_num_avg, &min_cores_needed);
  tpu_initialize();
  const int core_idx = tpu_core_index();
  if (core_idx < min_cores_needed) {
    // Shape:  [n',c',h',w'] ->  [1. slice_intracores, 1, size_from_axis]
    // Input:  [1, c, h, w]  ->  [1, c_sliced, 1, size_from_axis]
    // output: [1, c, h, w]  ->  [1, c_sliced, 1, size_from_axis]
    new_shape.c = outer_num_real;
    int sliced_shape[4];
    sliced_shape[0] = new_shape.n;
    sliced_shape[1] = new_shape.c;
    sliced_shape[2] = new_shape.h;
    sliced_shape[3] = new_shape.w;
    nodechip_softmax(api->input_global_addr + core_idx * outer_num_avg * new_shape.h * new_shape.w * tpu_data_type_size(api->dtype),
                     api->output_global_addr + core_idx * outer_num_avg * new_shape.h * new_shape.w * tpu_data_type_size(api->dtype),
                     sliced_shape, 4, 3, 3, 1, 1.f, (data_type_t)api->dtype);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_log_softmax_multi_core);
#endif