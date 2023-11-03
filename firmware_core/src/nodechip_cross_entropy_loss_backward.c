#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

void nodechip_cross_entropy_loss_backward (
global_addr_t input_global_addr,
global_addr_t target_global_addr,
global_addr_t weight_global_addr,
global_addr_t grad_output_global_addr,
global_addr_t grad_input_global_addr,
int batch_num,
int class_num,
int reduction,
float label_smoothing,
data_type_t dtype,
bool target_is_int64 )
{
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  //const scalar_t one_fp = tpu_fp_cast ( one_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  const scalar_t batch_num_inv_fp32 = { .f32 = 1.f / batch_num };
  const scalar_t batch_num_inv_fp = tpu_fp_cast ( batch_num_inv_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  const scalar_t onehot_hit_fp32 = { .f32 = ( 1.f - label_smoothing ) + label_smoothing / class_num };
  const scalar_t onehot_miss_fp32 = { .f32 = label_smoothing / class_num };
#if 0
  const scalar_t onehot_hit = tpu_fp_cast ( onehot_hit_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  const scalar_t onehot_miss = tpu_fp_cast ( onehot_miss_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN );
#endif
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  const dim4 input_shape = { .n = 1, .c = batch_num, .h = 1, .w = class_num };
  const dim4 target_shape = { .n = 1, .c = batch_num, .h = 1, .w = target_is_int64 ? 2 : 1 };
  int cmax = batch_num, wmax = class_num;
  dim4 input_global_stride, target_global_stride;
  tpu_continuous_stride ( &input_global_stride, &input_shape );
  tpu_continuous_stride ( &target_global_stride, &target_shape );
  local_addr_t next_base = 0;
  local_addr_t exp_coeff_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t exp_table_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t input_local_addrs[2], target_local_addr, grad_input_local_addrs[2];
  local_addr_t input_exp_local_addr;
  local_addr_t reduce_tile_local_addr, reduce_sum_local_addrs[2];
  local_addr_t sequence_local_addr, onehot_local_addr, weight_local_addrs[2];
  local_addr_t work0_local_addr, work1_local_addr;
  local_addr_t next;
  while ( true )
  {
    int size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, dtype );
    int size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, DT_FP32 );
    int reduce_tile_size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    int reduce_size_fp32 = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    //int reduce_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    int target_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, target_is_int64 ? 2 : 1, DT_INT32 );
    // int size_sequence = tpu_aligned_feature_size ( 1, wmax, DT_INT32 );
    // int onehot_size = tpu_aligned_feature_size ( 1, wmax, DT_FP32 );
    int weight_size = tpu_aligned_feature_size ( 1, wmax, dtype );
    next = next_base;
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    input_exp_local_addr = next; next += size_fp32;
    reduce_tile_local_addr = next; next += reduce_tile_size_fp32;
    reduce_sum_local_addrs[0] = next; next += reduce_size_fp32;
    reduce_sum_local_addrs[1] = next; next += reduce_size_fp32;
    target_local_addr = next; next += target_size;
    work0_local_addr = next; next += size_fp32;
    work1_local_addr = next; next += size_fp32;
    sequence_local_addr = work0_local_addr; // reuse
    onehot_local_addr = work1_local_addr; // reuse
    if ( weight_global_addr != 0 )
    {
      weight_local_addrs[0] = next; next += weight_size;
      weight_local_addrs[1] = next; next += weight_size;
    }
    grad_input_local_addrs[0] = next; next += size;
    grad_input_local_addrs[1] = next; next += size;
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
  scalar_t grad_output_scalar;
  tpu_invalidate_cache ( grad_output_global_addr, 64 );
  if ( dtype == DT_FP32 )
  {
    grad_output_scalar.u32 = * ( unsigned int * ) tpu_global_mem_addr ( grad_output_global_addr );
  }
  else if ( dtype != DT_FP32 || dtype == DT_BFP16 )
  {
    grad_output_scalar.u16 = * ( unsigned short * ) tpu_global_mem_addr ( grad_output_global_addr );
  }
  else
  {
    TPUKERNEL_ASSERT ( false );
  }
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
      if ( dtype != DT_FP32 )
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
    // Move target from global memory to local memory
    dim4 target_s2l_shape = { .n = 1, .c = shape.c, .h = 1, .w = target_is_int64 ? 2 : 1 };
    tpu_gdma_cpy_S2L ( target_local_addr, target_global_addr + ( 1UL * cdone * target_global_stride.c ) * sizeof ( int ), &target_s2l_shape, NULL, &target_global_stride, DT_INT32 );
    dim4 target_stride; tpu_aligned_stride ( &target_stride, 0, &reduce_shape, DT_INT32 );
    if ( target_is_int64 )
    {
      dim4 target_s2l_stride; tpu_aligned_stride ( &target_s2l_stride, 0, &target_s2l_shape, DT_INT32 );
      dim4 target_int64_stride = { .n = target_s2l_stride.n, .c = target_s2l_stride.c, .h = 1, .w = 1 };
      tpu_gdma_cpy_L2L ( target_local_addr, target_local_addr, &reduce_shape, NULL, &target_int64_stride, DT_INT32 );
    }
    wtodo = class_num, wdone = 0;
    if ( wmax == class_num )
    {
      index = 1 - index;
    }
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
      if ( weight_global_addr != 0 )
      {
        // Move weight from global memory to local memory
        dim4 weight_shape = { .n = 1, .c = 1, .h = 1, .w = shape.w };
        tpu_gdma_cpy_S2L ( weight_local_addrs[index], weight_global_addr + ( 1UL * wdone ) * dsize, &weight_shape, NULL, NULL, dtype );
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
        if ( dtype != DT_FP32 )
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
      // gread_input = input * input_sum
      dim4 reduce_stride; tpu_aligned_stride ( &reduce_stride, 0, &reduce_shape, DT_FP32 );
      dim4 reduce_bcast_stride = { .n = reduce_stride.n, .c = reduce_stride.c, 0, 0 };
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_fp_mul ( input_exp_local_addr, input_exp_local_addr, reduce_sum_local_addr, &tile_shape, NULL, NULL, &reduce_bcast_stride, DT_FP32 );
        tpu_bdc_cast ( grad_input_local_addrs[index], input_exp_local_addr, &tile_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
      }
      else
      {
        tpu_bdc_fp_mul ( grad_input_local_addrs[index], input_exp_local_addr, reduce_sum_local_addr, &tile_shape, NULL, NULL, &reduce_bcast_stride, dtype );
      }
      // gread_input = gread_input * weight
      if ( weight_global_addr != 0 )
      {
        dim4 weight_bcast_shape = { .n = 1, .c = MIN ( shape.c, NPU_NUM ), .h = 1, .w = shape.w };
        tpu_bdc_cpy_cross_npu ( weight_local_addrs[index], weight_local_addrs[index], &weight_bcast_shape, dtype );
        // input = input * weight
        dim4 weight_bcast_stride; tpu_aligned_stride ( &weight_bcast_stride, 0, &weight_bcast_shape, dtype );
        weight_bcast_stride.c = 0; // broadcast c
        tpu_bdc_fp_mul ( grad_input_local_addrs[index], grad_input_local_addrs[index], weight_local_addrs[index], &shape, NULL, NULL, &weight_bcast_stride, dtype );
      }
      // one-hot
      tpu_bdc_arithmetic_sequence_bcast ( sequence_local_addr, NPU_NUM, wdone, 1, shape.w );
      for ( int c = 0; c < DIV_UP ( shape.c, NPU_NUM ); ++c )
      {
        dim4 onehot_shape = { .n = 1, .c = MIN ( NPU_NUM, shape.c - c * NPU_NUM ), .h = shape.h, .w = shape.w };
        variable_t var_sequence = { .type = TENSOR, .context = { .addr = sequence_local_addr } };
        variable_t var_target = { .type = VECTOR, .context = { .addr = target_local_addr + c * target_stride.c * sizeof ( int ) } };
        variable_t var_hit = { .type = SCALAR, .context = { .scalar = onehot_hit_fp32 } };
        variable_t var_miss = { .type = SCALAR, .context = { .scalar = onehot_miss_fp32 } };
        tpu_bdc_equal_select ( onehot_local_addr, &var_sequence, &var_target, &var_hit, &var_miss, &onehot_shape, DT_INT32, DT_FP32 );
        if ( dtype != DT_FP32 )
        {
          tpu_bdc_cast ( onehot_local_addr, onehot_local_addr, &onehot_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
        }
        // gread_input = gread_input - one-hot
        tpu_bdc_fp_sub ( grad_input_local_addrs[index] + c * tile_stride.c * dsize, grad_input_local_addrs[index] + c * tile_stride.c * dsize, onehot_local_addr, &onehot_shape, NULL, NULL, NULL, dtype );
      }
      if ( reduction == 0 )
      {
        // gread_input = gread_input / batch_num
        tpu_bdc_fp_mul_C ( grad_input_local_addrs[index], grad_input_local_addrs[index], batch_num_inv_fp, &shape, NULL, NULL, dtype );
      }
      // grad_input = grad_input * grad_output_scalar
      tpu_bdc_fp_mul_C ( grad_input_local_addrs[index], grad_input_local_addrs[index], grad_output_scalar, &shape, NULL, NULL, dtype );
      // Move output from local memory to global memory
      dim4 stride = { .n = tile_stride.n, .c = tile_stride.c, .h = 1, .w = 1 };
      l2s = true;
      l2s_global_addr = grad_input_global_addr + ( cdone * input_global_stride.c + wdone * input_global_stride.w ) * dsize;
      l2s_local_addr = grad_input_local_addrs[index];
      l2s_shape = shape;
      l2s_global_stride = input_global_stride;
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

void tpu_kernel_api_cross_entropy_loss_backward ( const void * args )
{
  sg_api_cross_entropy_loss_backward_t * api = ( sg_api_cross_entropy_loss_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  TPUKERNEL_ASSERT ( api->reduction == 0 || api->reduction == 1 );
  tpu_initialize();
  nodechip_cross_entropy_loss_backward ( api->input_global_addr, api->target_global_addr, 0, api->grad_output_global_addr, api->grad_input_global_addr, api->batch, api->class_, api->reduction, api->label_smoothing, ( data_type_t ) api->dtype, api->is_target_int64 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_cross_entropy_loss_backward );