#include "sg_api_struct.h"
#include "tpu_kernel.h"

#if 0
void nodechip_cross_entropy_loss_forward (
global_addr_t input_global_addr,
global_addr_t target_global_addr,
global_addr_t weight_global_addr,
global_addr_t output_global_addr,
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
  const scalar_t one_fp = tpu_fp_cast ( one_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  const scalar_t batch_num_inv_fp32 = { .f32 = 1.f / batch_num };
  const scalar_t batch_num_inv_fp = tpu_fp_cast ( batch_num_inv_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  const scalar_t onehot_hit_fp32 = { .f32 = ( 1.f - label_smoothing ) + label_smoothing / class_num };
  const scalar_t onehot_miss_fp32 = { .f32 = label_smoothing / class_num };
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
  local_addr_t log_coeff_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t exp_coeff_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t exp_table_addr = next_base; next_base += tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t input_local_addrs[2], target_local_addr, output_local_addrs[2];
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
    int reduce_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    int target_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, target_is_int64 ? 2 : 1, DT_INT32 );
    int weight_size = tpu_aligned_feature_size ( 1, wmax, dtype );
    int size_cw_trans = tpu_aligned_feature_size ( 1, MIN ( cmax, NPU_NUM ), dtype );
    next = next_base;
    output_local_addrs[0] = next; next += MAX ( reduce_size, size_cw_trans );
    output_local_addrs[1] = next; next += reduce_size;
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
  tpu_bdc_load_fp32_log_coeff ( log_coeff_addr );
  tpu_bdc_load_fp32_exp_coeff ( exp_coeff_addr );
  tpu_bdc_load_fp32_exp_table ( exp_table_addr );
  dim4 shape = { .n = 1, .h = 1 };
  // set output zeros
  dim4 max_reduce_shape = { .n = 1, .c = cmax, .h = 1, .w = 1 };
  dim4 max_reduce_stride; tpu_aligned_stride ( &max_reduce_stride, 0, &max_reduce_shape, dtype );
  tpu_bdc_set_C ( output_local_addrs[1], zero, &max_reduce_shape, NULL, dtype );
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
    // input_sum = log ( input_sum )
    tpu_bdc_fp32_log ( reduce_sum_local_addr, reduce_sum_local_addr, work0_local_addr, log_coeff_addr, &reduce_shape );
    // input_sum FP32 -> FP16 ( inplace )
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast ( reduce_sum_local_addr, reduce_sum_local_addr, &reduce_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
    }
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
      // input = input_sum - input = log ( sum ( exp ( input ) ) ) - input
      dim4 reduce_stride; tpu_aligned_stride ( &reduce_stride, 0, &reduce_shape, dtype );
      dim4 reduce_bcast_stride = { .n = reduce_stride.n, .c = reduce_stride.c, 0, 0 };
      tpu_bdc_fp_sub ( input_local_addrs[index], reduce_sum_local_addr, input_local_addrs[index], &tile_shape, NULL, &reduce_bcast_stride, NULL, dtype );
      // input = input * weight
      if ( weight_global_addr != 0 )
      {
        dim4 weight_bcast_shape = { .n = 1, .c = MIN ( shape.c, NPU_NUM ), .h = 1, .w = shape.w };
        tpu_bdc_npu_bcast ( weight_local_addrs[index], weight_local_addrs[index], &weight_bcast_shape, dtype );
        // input = input * weight
        dim4 weight_bcast_stride; tpu_aligned_stride ( &weight_bcast_stride, 0, &weight_bcast_shape, dtype );
        weight_bcast_stride.c = 0; // broadcast c
        tpu_bdc_fp_mul ( input_local_addrs[index], input_local_addrs[index], weight_local_addrs[index], &shape, NULL, NULL, &weight_bcast_stride, dtype );
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
        // input = input * one-hot
        tpu_bdc_fp_mul ( input_local_addrs[index] + c * tile_stride.c * dsize, input_local_addrs[index] + c * tile_stride.c * dsize, onehot_local_addr, &onehot_shape, NULL, NULL, NULL, dtype );
      }
      // set input tail
      if ( shape.w % tile != 0 )
      {
        dim4 tail_shape = { .n = 1, .c = shape.c, .h = 1, .w = tile - ( shape.w % tile ) };
        tpu_bdc_set_C ( input_local_addrs[index] + shape.w * dsize, zero, &tail_shape, &tile_stride, dtype );
      }
      // [ 1, shape.c, DIV_UP ( shape.w, tile ), tile ] -> [ 1, shape.c, 1, tile ] -> [ 1, shape.c, 1, 1 ]
      dim2 kernel = { .h = tile_shape.h, .w = 1 };
      tpu_bdc_fp_avg_pool2d ( reduce_tile_local_addr, input_local_addrs[index], &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, one_fp );
      tile_shape.h = 1; kernel.h = 1; kernel.w = tile_shape.w;
      tpu_bdc_fp_avg_pool2d ( output_local_addrs[0], reduce_tile_local_addr, &tile_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, one_fp );
      // output_1 = output_1 + output_0
      tpu_bdc_fp_add ( output_local_addrs[1], output_local_addrs[1], output_local_addrs[0], &reduce_shape, NULL, NULL, NULL, dtype );
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
  for ( int c = 1; c < DIV_UP ( cmax, NPU_NUM ); ++c )
  {
    dim4 sum_shape = { .n = 1, .c = MIN ( NPU_NUM, cmax - c * NPU_NUM ), .h = 1, .w = 1 };
    tpu_bdc_fp_add ( output_local_addrs[1], output_local_addrs[1], output_local_addrs[1] + c * max_reduce_stride.c * dsize, &sum_shape, NULL, NULL, NULL, dtype );
  }
  // cw trans
  dim4 cw_trans_shape = { .n = 1, .c = 1, .h = 1, .w = MIN ( cmax, NPU_NUM ) };
  tpu_bdc_wc_trans ( output_local_addrs[0], output_local_addrs[1], &cw_trans_shape, dtype );
  dim2 kernel = { .h = cw_trans_shape.h, .w = cw_trans_shape.w };
  tpu_bdc_fp_avg_pool2d ( output_local_addrs[1], output_local_addrs[0], &cw_trans_shape, &kernel, &zero_pad, &stride_one, &dilation_one, dtype, reduction == 0 ? batch_num_inv_fp : one_fp );
  dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  tpu_gdma_cpy_L2S ( output_global_addr, output_local_addrs[1], &output_shape, NULL, NULL, dtype );
}
#else
inline static void pipeline_move(int *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}
void one_hot_core(
    local_addr_t input_addr,
    local_addr_t target_addr,
    local_addr_t sequence_table,
    local_addr_t buffer_addr,     // (NPU_NUM, shape.w)
    local_addr_t output_addr,
    dim4* shape,
    int offset,
    float hit,
    float miss,
    data_type_t dtype) {
  dim4 s_shape = {1, NPU_NUM, 1, shape->w};
  if (offset != 0) {
    scalar_t C = {.s32 = offset};
    tpu_bdc_int_add_C(
        buffer_addr,
        sequence_table,
        C,
        &s_shape,
        NULL, NULL,
        DT_INT32, DT_INT32, DT_INT32,
        0,
        RM_HALF_AWAY_FROM_ZERO,
        false);
  }
  const scalar_t onehot_hit_fp32 = {.f32 = hit};
  // const scalar_t onehot_hit_fp = tpu_fp_cast(onehot_hit_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN);
  const scalar_t onehot_miss_fp32 = {.f32 = miss};
  // const scalar_t onehot_miss_fp = tpu_fp_cast(onehot_miss_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN);
  int c_stride = tpu_aligned_feature_size(1, 1, DT_INT32);
  dim4 stride = {0}, stride_fp32 = {0};
  tpu_aligned_stride(&stride, 0, shape, dtype);
  tpu_aligned_stride(&stride_fp32, 0, shape, DT_FP32);
  for (int c = 0; c < DIV_UP(shape->c, NPU_NUM); ++c) {
    dim4 one_hot_shape = {1, MIN(NPU_NUM, shape->c - c * NPU_NUM), 1, shape->w};
    variable_t var_sequence = {.type = TENSOR, .context = {.addr = offset == 0 ? sequence_table : buffer_addr}};
    variable_t var_target = {.type = VECTOR, .context = {.addr = target_addr + c * c_stride}};
    variable_t var_hit = {.type = SCALAR, .context = {.scalar = onehot_hit_fp32}};
    variable_t var_miss = {.type = SCALAR, .context = {.scalar = onehot_miss_fp32}};
    tpu_bdc_equal_select(
        output_addr + stride_fp32.c * c * tpu_data_type_size(DT_FP32),
        &var_sequence,
        &var_target,
        &var_hit,
        &var_miss,
        &one_hot_shape,
        DT_INT32, DT_FP32);
    if (dtype != DT_FP32) {
      tpu_bdc_cast(
          output_addr + stride.c * c * tpu_data_type_size(dtype),
          output_addr + stride_fp32.c * c * tpu_data_type_size(DT_FP32),
          &one_hot_shape,
          &stride, &stride_fp32,
          dtype, DT_FP32,
          RM_HALF_TO_EVEN);
    }
    tpu_bdc_fp_mul(
        output_addr + stride.c * c * tpu_data_type_size(dtype),
        input_addr + stride.c * c * tpu_data_type_size(dtype),
        output_addr + stride.c * c * tpu_data_type_size(dtype),
        &one_hot_shape,
        &stride, &stride, &stride,
        dtype);
  }
}

void sum_core(
    local_addr_t input_addr,
    local_addr_t buffer_addr,
    local_addr_t output_addr,
    dim4 *shape,
    bool add_result,
    data_type_t dtype) {
  const int eu_num = tpu_eu_num(dtype);
  scalar_t zero_fp = {.u32 = 0};
  if (shape->w % eu_num != 0) {
    dim4 tail_shape = {1, shape->c, 1, eu_num - (shape->w % eu_num)};
    dim4 tile_shape = {.n = 1, .c = shape->c, .h = DIV_UP (shape->w, eu_num), .w = eu_num};
    dim4 tile_stride; tpu_aligned_stride (&tile_stride, 0, &tile_shape, dtype);
    tpu_bdc_set_C(
        input_addr + shape->w * tpu_data_type_size(dtype),
        zero_fp,
        &tail_shape,
        &tile_stride,
        dtype);
  }
  const dim2 stride_one = {.h = 1, .w = 1};
  const dim2 dilation_one = {.h = 1, .w = 1};
  const padding_t zero_pad = {.top = 0, .left = 0, .bottom = 0, .right = 0};
  const scalar_t one_fp = {.f32 = 1.f};
  dim4 avg_shape = {.n = 1, .c = shape->c, .h = DIV_UP(shape->w, eu_num), .w = eu_num};
  dim2 kernel = {avg_shape.h, 1};
  tpu_bdc_fp_avg_pool2d(
      add_result ? buffer_addr : output_addr,
      input_addr,
      &avg_shape,
      &kernel,
      &zero_pad,
      &stride_one,
      &dilation_one,
      dtype,
      tpu_cast(one_fp, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
  if (add_result) {
    dim4 sum_shape = {.n = 1, .c = shape->c, .h = 1, .w = eu_num};
    tpu_bdc_fp_add(
        output_addr,
        output_addr,
        buffer_addr,
        &sum_shape,
        NULL, NULL, NULL,
        dtype);
  }
}

void nodechip_coress_entropy_local(
    local_addr_t input_addr,
    local_addr_t target_addr,
    local_addr_t work0_addr,
    local_addr_t work1_addr,
    local_addr_t work2_addr,    // can same with input
    local_addr_t exp_coeff_addr,
    local_addr_t sequence_table,
    local_addr_t one_hot_sum_addr,
    local_addr_t exp_sum_addr,
    dim4* shape,
    int offset,
    int is_first,
    float hit,
    float miss,
    data_type_t dtype) {

  // one hot
  one_hot_core(
      input_addr,
      target_addr,
      sequence_table,
      work0_addr,
      work1_addr,
      shape,
      offset,
      hit,
      miss,
      dtype);
  // SUM0 = sum(one_hot)
  sum_core(work1_addr, work0_addr, one_hot_sum_addr, shape, is_first == false, dtype);
  // EXP = exp(x)
  tpu_bdc_fp_exp(
      work2_addr,
      input_addr,
      work0_addr,
      work1_addr,
      exp_coeff_addr,
      shape,
      dtype);
  // SUM = sum(EXP)
  sum_core(work2_addr, work0_addr, exp_sum_addr, shape, offset != 0, dtype);
}

// exp_coeff, log_coeff, sequence stay in local mem
/* +-----------+-----------+----------+---------+-------------+---------+--------+--------+--------+-------+-------+
 * | exp_coeff | log_coeff | sequence | exp_sum | one_hot_sum | log_exp | target | input0 | input1 | work0 | work1 |
 * +-----------+-----------+----------+---------+-------------+---------+--------+--------+--------+-------+-------+
*/
typedef struct {
  local_addr_t input_addr[2];
  local_addr_t target_addr;
  local_addr_t exp_coeff_addr;
  local_addr_t log_coeff_addr;
  local_addr_t seq_addr;
  local_addr_t work0_addr;
  local_addr_t work1_addr;
  local_addr_t exp_sum_addr;
  local_addr_t one_hot_sum_addr;
  local_addr_t log_sum_addr;
} loc_scheme_t;
void cal_cw_split(int c, int w, int* cw_split, loc_scheme_t* sm, data_type_t dtype) {
  const int eu_num = tpu_eu_num(dtype);
  int c_split = c, w_split = w;
  while (true) {
    int w_size = tpu_aligned_feature_size(1, w_split, dtype);
    int c_per_npu = tpu_channle_num_per_npu(0, c_split);
    int coeff_size = ALIGN(32 * sizeof(float), ALIGN_BYTES);
    int sequence_size = tpu_aligned_feature_size(1, w_split, DT_INT32);
    int input_size = c_per_npu * w_size;
    int work1_size = c_per_npu * tpu_aligned_feature_size(1, w_split, DT_FP32);
    int target_size = c_per_npu * tpu_aligned_feature_size(1, 1, DT_INT32);
    int log_size = c_per_npu * tpu_aligned_feature_size(1, 1, dtype);
    int sum_size = c_per_npu * tpu_aligned_feature_size(1, eu_num, dtype);
    int mem_size = ALIGN(coeff_size * 2 + sequence_size + sum_size * 2 + target_size + log_size, BANK_SIZE);
    mem_size += 3 * ALIGN(input_size, BANK_SIZE);
    mem_size += ALIGN(work1_size, BANK_SIZE);
    if (mem_size <= tpu_local_mem_size_per_npu()) {
      sm->exp_coeff_addr = 0;
      sm->log_coeff_addr = sm->exp_coeff_addr + coeff_size;
      sm->seq_addr = sm->log_coeff_addr + coeff_size;
      sm->exp_sum_addr = sm->seq_addr + sequence_size;
      sm->one_hot_sum_addr = sm->exp_sum_addr + sum_size;
      sm->log_sum_addr = sm->one_hot_sum_addr + sum_size;
      sm->target_addr = sm->log_sum_addr + log_size;
      sm->input_addr[0] = ALIGN(sm->target_addr + target_size, BANK_SIZE);
      sm->input_addr[1] = ALIGN(sm->input_addr[0] + input_size, BANK_SIZE);
      sm->work0_addr = ALIGN(sm->input_addr[1] + input_size, BANK_SIZE);
      sm->work1_addr = ALIGN(sm->work0_addr + input_size, BANK_SIZE);
      break;
    } else if (c_split > NPU_NUM) {
      c_split -= (c_split % NPU_NUM == 0 ? NPU_NUM : c_split % NPU_NUM);
    } else if (w_split > eu_num) {
      w_split -= (w_split % eu_num == 0 ? eu_num : w_split % eu_num);
    } else {
      TPUKERNEL_ASSERT(false);
    }
  }
  cw_split[0] = c_split;
  cw_split[1] = w_split;
  return;
}

void nodechip_cross_entropy_loss_forward(
    global_addr_t input_global_addr,
    global_addr_t target_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t output_global_addr,
    int batch_num,
    int class_num,
    int reduction,
    float label_smoothing,
    data_type_t in_dtype,
    bool target_is_int64) {

  TPUKERNEL_ASSERT(weight_global_addr == 0);
  data_type_t dtype = DT_FP32;
  const int eu_num = tpu_eu_num(dtype);
  const float hit = ( 1.f - label_smoothing ) + label_smoothing / class_num;
  const float miss = label_smoothing / class_num;
  const int step = target_is_int64 ? 2 : 1;
  const scalar_t batch_num_inv_fp32 = {.f32 = 1.f / batch_num};
  const scalar_t batch_num_inv_fp = tpu_fp_cast(batch_num_inv_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN);

  // data split
  loc_scheme_t loc_sm = {0};
  int cw_split[2] = {0};
  cal_cw_split(batch_num, class_num, cw_split, &loc_sm, DT_FP32);
  int c_split = cw_split[0], w_split = cw_split[1];

  // init
  tpu_bdc_load_fp_exp_coeff(loc_sm.exp_coeff_addr, dtype);
  tpu_bdc_load_fp32_log_coeff(loc_sm.log_coeff_addr);
  tpu_bdc_arithmetic_sequence_bcast(loc_sm.seq_addr, NPU_NUM, 0, 1, w_split);
  const dim2 stride_one = {.h = 1, .w = 1};
  const dim2 dilation_one = {.h = 1, .w = 1};
  const padding_t zero_pad = {.top = 0, .left = 0, .bottom = 0, .right = 0};
  const scalar_t one_fp32 = {.f32 = 1.f};
  const scalar_t one_fp = tpu_fp_cast(one_fp32, dtype, DT_FP32, RM_HALF_TO_EVEN);
  dim4 gshape = {.n = 1, .c = batch_num, .h =1, .w = class_num};
  dim4 gstride; tpu_continuous_stride(&gstride, &gshape);

  int cur_c_idx = 0, cur_c_len = 0;
  while (cur_c_idx < batch_num)
  {
    cur_c_len = MIN(batch_num - cur_c_idx, c_split);
    dim4 target_shape = {.n = 1, .c = cur_c_len, .h = 1, .w = step};
    tpu_gdma_cpy_S2L(
        loc_sm.target_addr,
        target_global_addr + cur_c_idx * step * tpu_data_type_size(DT_INT32),
        &target_shape,
        NULL, NULL,
        dtype);
    int cur_w_idx[3] = {0}, cur_w_len[3] = {0};
    int stage_idx = 0, draning_idx = 0;
    while (cur_w_idx[2] < class_num) {
      if (stage_idx > 0) {
        tpu_parallel_start();
      }
      // update load info
      if (draning_idx < 1) {
        cur_w_len[0] = MIN(class_num - cur_w_idx[0], w_split);
      }

      // load input
      if (draning_idx < 1) {
        dim4 shape = {.n = 1, .c = cur_c_len, .h = 1, .w = cur_w_len[0]};
        tpu_gdma_cpy_S2L(
            loc_sm.input_addr[stage_idx & 0x1],
            input_global_addr + (cur_c_idx * class_num + cur_w_idx[0]) * tpu_data_type_size(in_dtype),
            &shape,
            NULL, &gstride,
            in_dtype);
      }

      // compute
      if (stage_idx > 0 && draning_idx < 2) {
        dim4 shape = {.n = 1, .c = cur_c_len, .h = 1, .w = cur_w_len[1]};
        local_addr_t input_addr = loc_sm.input_addr[(stage_idx + 1) & 0x01];
        local_addr_t work0_addr = loc_sm.work0_addr;
        if (in_dtype != dtype) {
          tpu_bdc_cast(
              work0_addr,
              input_addr,
              &shape,
              NULL, NULL,
              dtype, in_dtype,
              RM_HALF_AWAY_FROM_ZERO);
          work0_addr = loc_sm.input_addr[(stage_idx + 1) & 0x01];
          input_addr = loc_sm.work0_addr;
        }
        nodechip_coress_entropy_local(
            input_addr,
            loc_sm.target_addr,
            work0_addr,
            loc_sm.work1_addr,
            input_addr,
            loc_sm.exp_coeff_addr,
            loc_sm.seq_addr,
            loc_sm.one_hot_sum_addr,
            loc_sm.exp_sum_addr,
            &shape,
            cur_w_idx[1],
            cur_c_idx == 0 && cur_w_idx[1] == 0,
            hit,
            miss,
            dtype);
      }

      if (tpu_is_parallel_state()) {
        tpu_parallel_end();
      }
      pipeline_move(cur_w_idx, 3);
      pipeline_move(cur_w_len, 3);
      if (draning_idx < 1) {
        cur_w_idx[0] += cur_w_len[0];
        if (cur_w_idx[0] >= class_num) {
          draning_idx++;
        }
      } else {
        draning_idx++;
      }
      stage_idx++;
    }
    tpu_parallel_start();
    dim4 exp_shape = {.n = 1, .c = cur_c_len, .h = 1, .w = eu_num};
    dim2 kernel = {.h = 1, .w = eu_num};
    tpu_bdc_fp_avg_pool2d(
        loc_sm.work0_addr,
        loc_sm.exp_sum_addr,
        &exp_shape,
        &kernel,
        &zero_pad,
        &stride_one,
        &dilation_one,
        dtype,
        one_fp);
    target_shape.w = 1;
    tpu_bdc_fp32_log(
        cur_c_idx == 0 ? loc_sm.log_sum_addr : loc_sm.input_addr[1],
        loc_sm.work0_addr,
        loc_sm.work1_addr,
        loc_sm.log_coeff_addr,
        &target_shape);
    if (cur_c_idx > 0) {
      dim4 sum_shape = {.n = 1, .c = cur_c_len, .h = 1, .w = 1};
      tpu_bdc_fp_add(
          loc_sm.log_sum_addr,
          loc_sm.log_sum_addr,
          loc_sm.input_addr[1],
          &sum_shape,
          NULL, NULL, NULL,
          dtype);
    }
    cur_c_idx += cur_c_len;
  }
  if (tpu_is_parallel_state()) {
    tpu_parallel_end();
  }

  dim4 one_hot_shape = {.n = 1, .c = c_split, .h = 1, .w = eu_num};
  dim2 kernel = {.h = 1, .w = eu_num};
  tpu_bdc_fp_avg_pool2d(
      loc_sm.work1_addr,
      loc_sm.one_hot_sum_addr,
      &one_hot_shape,
      &kernel,
      &zero_pad,
      &stride_one,
      &dilation_one,
      dtype,
      one_fp);
  tpu_bdc_fp_sub(
      loc_sm.work0_addr,
      loc_sm.log_sum_addr,
      loc_sm.work1_addr,
      &one_hot_shape,
      NULL, NULL, NULL,
      dtype);
  for ( int c = 1; c < DIV_UP(c_split, NPU_NUM); ++c ) {
    dim4 sum_shape = {.n = 1, .c = MIN(NPU_NUM, c_split - c * NPU_NUM), .h = 1, .w = 1};
    tpu_bdc_fp_add (
        loc_sm.work0_addr,
        loc_sm.work0_addr,
        loc_sm.work0_addr + c * tpu_aligned_feature_size(1, 1, dtype),
        &sum_shape,
        NULL, NULL, NULL,
        dtype);
  }
  // cw trans
  dim4 cw_trans_shape = {.n = 1, .c = 1, .h = 1, .w = MIN(c_split, NPU_NUM)};
  tpu_bdc_wc_trans(loc_sm.work1_addr, loc_sm.work0_addr, &cw_trans_shape, dtype);
  kernel.h = cw_trans_shape.h;
  kernel.w = cw_trans_shape.w;
  tpu_bdc_fp_avg_pool2d(
      loc_sm.work0_addr,
      loc_sm.work1_addr,
      &cw_trans_shape,
      &kernel,
      &zero_pad,
      &stride_one,
      &dilation_one,
      dtype,
      reduction == 0 ? batch_num_inv_fp : one_fp
  );
  dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  if (dtype != in_dtype) {
    tpu_bdc_cast(
        loc_sm.work1_addr,
        loc_sm.work0_addr,
        &output_shape,
        NULL, NULL,
        in_dtype, dtype,
        RM_HALF_AWAY_FROM_ZERO);
  }
  tpu_gdma_cpy_L2S(
      output_global_addr,
      dtype != in_dtype ? loc_sm.work1_addr : loc_sm.work0_addr,
      &output_shape,
      NULL, NULL,
      in_dtype);
}
#endif

void tpu_kernel_api_cross_entropy_loss ( const void * args )
{
  sg_api_cross_entropy_loss_t * api = ( sg_api_cross_entropy_loss_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  TPUKERNEL_ASSERT ( api->reduction == 0 || api->reduction == 1 );
  tpu_initialize();
  nodechip_cross_entropy_loss_forward ( api->input_global_addr,
                                        api->target_global_addr,
                                        0,
                                        api->output_global_addr,
                                        api->batch,
                                        api->class_,
                                        api->reduction,
                                        api->label_smoothing,
                                        ( data_type_t ) api->dtype,
                                        api->is_target_int64 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_cross_entropy_loss );
