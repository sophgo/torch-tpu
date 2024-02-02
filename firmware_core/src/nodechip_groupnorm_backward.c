#include "sg_api_struct.h"
#include "tpu_kernel.h"


#define DEFAULT_LOCAL_ADDR 0xFFFFFFFF
#define MIN(x,y) (((x)) < ((y)) ? (x) : (y))
#define GDMA_MAX_N 65535
#define GDMA_MAX_C 65535
#define GDMA_MAX_H 65535
#define GDMA_MAX_W 65535
#define GDMA_VALUE_DIR_S2S 2

typedef u32 stride_type;


// Input Normalized = ( Input − Saved Mean ) × Saved Invstd
// Grad Weight = ∑<(c//g), h, w> Grad Output × Input Normalized
// Grad Bias = ∑<(c//g), h, w> Grad Output
// Grad Input = Weight × Saved Invstd × ( Grad Output − ( Input Normalized × Grad Weight + Grad Bias ) / (C//G)HW )
//                         Grad Output       Input       Weight       Mean       Invstd       Input Normalized       Grad Weight       Grad Bias
// Input Normalized        X                 O           X            O          O            SELF                   X                 X
// grad_input_enable       O                 X           O            X          O            O                      O                 O
// grad_weight_enable      O                 X           X            X          X            O                      SELF              X
// grad_bias_enable        O                 X           X            X          X            X                      X                 SELF

static inline bool nodechip_groupnorm2d_backward_split_c_only_all_fp32 (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t saved_mean_global_addr,
global_addr_t saved_invstd_global_addr,
global_addr_t grad_input_global_addr,
global_addr_t grad_weight_global_addr,
global_addr_t grad_bias_global_addr,
dim4 shape,
data_type_t dtype,
int grad_input_enable,
int grad_weight_enable,
int grad_bias_enable,
int group_nums )
{
  TPUKERNEL_ASSERT(shape.c % group_nums == 0);
  dim4 grouped_shape = {.n = shape.n, .c = group_nums, .h = (shape.c / group_nums) * shape.h, .w = shape.w};
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  dim4 input_global_stride; tpu_continuous_stride ( &input_global_stride, &grouped_shape );
  dim4 grad_output_global_stride; tpu_continuous_stride ( &grad_output_global_stride, &grouped_shape );
  dim4 grad_input_global_stride; tpu_continuous_stride ( &grad_input_global_stride, &grouped_shape );
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  const scalar_t neg_inv_cghw_fp32 = { .f32 = -1.0 / ( 1UL * grouped_shape.h * grouped_shape.w ) };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  local_addr_t input_local_addrs[2] = { DEFAULT_LOCAL_ADDR }, grad_output_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t input_normalized_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t saved_mean_local_addrs[2] = { DEFAULT_LOCAL_ADDR }, saved_invstd_local_addrs[2] = { DEFAULT_LOCAL_ADDR }, weight_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t saved_mean_fp32_local_addr = DEFAULT_LOCAL_ADDR, saved_invstd_fp32_local_addr = DEFAULT_LOCAL_ADDR, weight_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_output_reduce_tile_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_output_reduce_tile_n_piece_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_bias_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t grad_output_input_normalized_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_output_input_normalized_reduce_tile_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_output_input_normalized_reduce_tile_n_piece_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_weight_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t grad_output_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_bias_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_weight_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_input_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t next = 0;
  int cmax = grouped_shape.c;
  while ( true )
  {
    int tile_size = shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( grouped_shape.h * grouped_shape.w, tile ), tile, dtype );  // h*w*4
    int tile_fp32_size = shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( grouped_shape.h * grouped_shape.w, tile ), tile, DT_FP32 );   // h*w*4
    int nc_size = grouped_shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( 1, tile ), tile, dtype );
    int nc_fp32_size = grouped_shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( 1, tile ), tile, dtype );  // 64
    int reduce_tile_fp32_size = shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    int reduce_tile_n_piece_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );   //  64
    int reduce_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 ); // 64
    int grad_bias_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );   // 64
    int grad_weight_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );   // 64
    next = 0;
    input_local_addrs[0] = next; next += tile_size;
    input_local_addrs[1] = next; next += tile_size;
    grad_output_local_addrs[0] = next; next += tile_size;
    grad_output_local_addrs[1] = next; next += tile_size;
    input_normalized_fp32_local_addr = next; next += tile_fp32_size;
    saved_mean_local_addrs[0] = next; next += nc_size;
    saved_mean_local_addrs[1] = next; next += nc_size;
    if ( dtype != DT_FP32 )
    {
      saved_mean_fp32_local_addr = next; next += nc_fp32_size;
    }
    saved_invstd_local_addrs[0] = next; next += nc_size;
    saved_invstd_local_addrs[1] = next; next += nc_size;
    if ( dtype != DT_FP32 )
    {
      saved_invstd_fp32_local_addr = next; next += nc_fp32_size;
    }
    if ( grad_input_enable )
    {
      grad_input_local_addrs[0] = next; next += tile_size;
      grad_input_local_addrs[1] = next; next += tile_size;
    }
    if ( grad_input_enable || grad_bias_enable )
    {
      grad_output_reduce_tile_fp32_local_addr = next; next += reduce_tile_fp32_size;
      grad_output_reduce_tile_n_piece_fp32_local_addr = next; next += reduce_tile_n_piece_fp32_size;
      if ( dtype != DT_FP32 )
      {
        grad_bias_fp32_local_addr = next; next += reduce_fp32_size;
      }
      grad_bias_local_addrs[0] = next; next += grad_bias_size;
      grad_bias_local_addrs[1] = next; next += grad_bias_size;
    }
    if ( grad_weight_enable || grad_input_enable )
    {
      grad_output_input_normalized_fp32_local_addr = next; next += tile_fp32_size;
      grad_output_input_normalized_reduce_tile_fp32_local_addr = next; next += reduce_tile_fp32_size;
      grad_output_input_normalized_reduce_tile_n_piece_fp32_local_addr = next; next += reduce_tile_n_piece_fp32_size;
      if ( dtype != DT_FP32 )
      {
        grad_weight_fp32_local_addr = next; next += reduce_fp32_size;
      }
      grad_weight_local_addrs[0] = next; next += grad_weight_size;
      grad_weight_local_addrs[1] = next; next += grad_weight_size;
    }
    if ( dtype != DT_FP32 )
    {
      grad_output_fp32_local_addr = next; next += tile_fp32_size;
    }
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
  bool l2s_grad_bias = false, l2s_grad_weight = false, l2s_grad_input = false;
  local_addr_t l2s_grad_bias_local_addr = DEFAULT_LOCAL_ADDR, l2s_grad_weight_local_addr = DEFAULT_LOCAL_ADDR, l2s_grad_input_local_addr = DEFAULT_LOCAL_ADDR;
  global_addr_t l2s_grad_bias_global_addr = DEFAULT_LOCAL_ADDR, l2s_grad_weight_global_addr = DEFAULT_LOCAL_ADDR, l2s_grad_input_global_addr = DEFAULT_LOCAL_ADDR;
  dim4 l2s_grad_bias_shape, l2s_grad_weight_shape, l2s_grad_input_shape;
  dim4 work_shape = { .n = grouped_shape.n, .h = grouped_shape.h, .w = grouped_shape.w };
  int index = 0;
  int ctodo = grouped_shape.c, cdone = 0; // cdone : done channels
  while ( ctodo != 0 ) // loop per NPU_NUM
  {
    work_shape.c = MIN ( ctodo, cmax );
    dim4 channel_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = 1 };
    dim4 channel_stride; tpu_aligned_stride ( &channel_stride, 0, &channel_shape, dtype );
    channel_stride.h = 0, channel_stride.w = 0;
    dim4 reduce_tile_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = tile };
    dim4 reduce_tile_fp32_stride; tpu_aligned_stride ( &reduce_tile_fp32_stride, 0, &reduce_tile_shape, DT_FP32 );
    dim4 reduce_tile_n_piece_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = tile };
    dim4 reduce_shape = { .n = 1, .c = work_shape.c, .h = 1, .w = 1 };
    dim2 reduce_kernel = { .h = 1, .w = tile };
    dim4 tile_shape = { .n = work_shape.n, .c = work_shape.c, .h = DIV_UP ( ( work_shape.h * work_shape.w ), tile ), .w = tile };
    dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
    dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
    dim4 tail_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = tile - ( ( work_shape.h * work_shape.w ) % tile ) };
    dim2 reduce_tile_kernel = { .h = tile_shape.h, .w = 1 };
    // Move Saved Mean from global to local memory
    tpu_gdma_cpy_S2L ( saved_mean_local_addrs[index],
                       saved_mean_global_addr + 1UL * cdone * dsize,
                       &channel_shape,
                       NULL,
                       NULL,
                       dtype );
    // Move Saved Invstd from global to local memory
    tpu_gdma_cpy_S2L ( saved_invstd_local_addrs[index],
                       saved_invstd_global_addr + 1UL * cdone * dsize,
                       &channel_shape,
                       NULL,
                       NULL,
                       dtype );
    // Move Input from global memory to local memory
    tpu_gdma_cpy_S2L ( input_local_addrs[index],
                       input_global_addr + 1UL * cdone * input_global_stride.c * dsize,
                       &work_shape,
                       NULL, // tile is eu num of dtype, so NULL is OK
                       &input_global_stride,
                       dtype );
    // Move Grad Output from global memory to local memory
    tpu_gdma_cpy_S2L ( grad_output_local_addrs[index],
                       grad_output_global_addr + 1UL * cdone * grad_output_global_stride.c * dsize,
                       &work_shape,
                       NULL, // tile is eu num of dtype, so NULL is OK
                       &grad_output_global_stride,
                       dtype );
    // Synchronize Point
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    // Move Grad Bias from local memory to global memory  
    if ( l2s_grad_bias )
    {
      tpu_gdma_cpy_L2S ( l2s_grad_bias_global_addr,
                         l2s_grad_bias_local_addr,
                         &l2s_grad_bias_shape,
                         NULL,
                         NULL,
                         dtype );
      l2s_grad_bias = false;
    }
    // Move Grad Weight from local memory to global memory   
    if ( l2s_grad_weight )
    {
      tpu_gdma_cpy_L2S ( l2s_grad_weight_global_addr,
                         l2s_grad_weight_local_addr,
                         &l2s_grad_weight_shape,
                         NULL,
                         NULL,
                         dtype );
      l2s_grad_weight = false;
    }
    // Move Grad Input from local memory to global memory  
    if ( l2s_grad_input )
    {
      tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                         l2s_grad_input_local_addr,
                         &l2s_grad_input_shape,
                         &grad_input_global_stride,
                         NULL,
                         dtype );
      l2s_grad_input = false;
    }
    // Set ∑<c//g ( partial ), h, w> Grad Output zeros
    if ( grad_input_enable || grad_bias_enable )
    {
      tpu_bdc_set_C ( grad_output_reduce_tile_n_piece_fp32_local_addr,
                      zero,
                      &reduce_tile_n_piece_shape,
                      NULL,
                      DT_FP32 );
    }
    // Set ∑<c//g ( partial ), h, w> Grad Output x Intpu Normalized zeros
    if ( grad_input_enable || grad_weight_enable )
    {
      tpu_bdc_set_C ( grad_output_input_normalized_reduce_tile_n_piece_fp32_local_addr,
                      zero,
                      &reduce_tile_n_piece_shape,
                      NULL,
                      DT_FP32 );
    }
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast ( saved_mean_fp32_local_addr,
                     saved_mean_local_addrs[index],
                     &channel_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
      tpu_bdc_cast ( saved_invstd_fp32_local_addr,
                     saved_invstd_local_addrs[index],
                     &channel_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
    }
    else
    {
      saved_mean_fp32_local_addr = saved_mean_local_addrs[index];
      saved_invstd_fp32_local_addr = saved_invstd_local_addrs[index];
    }
    // Input Normalized = ( Input − Saved Mean ) × Saved Invstd
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast ( input_normalized_fp32_local_addr,
                     input_local_addrs[index],
                     &tile_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
      tpu_bdc_fp_sub ( input_normalized_fp32_local_addr,
                       input_normalized_fp32_local_addr,
                       saved_mean_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
      tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                       input_normalized_fp32_local_addr,
                       saved_invstd_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
    }
    else
    {
      tpu_bdc_fp_sub ( input_normalized_fp32_local_addr,
                       input_local_addrs[index],
                       saved_mean_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
      tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                       input_normalized_fp32_local_addr,
                       saved_invstd_local_addrs[index],
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
    }
    // Grad Bias
    if ( grad_input_enable || grad_bias_enable )
    {
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_cast ( grad_output_fp32_local_addr,
                       grad_output_local_addrs[index],
                       &tile_shape,
                       NULL,
                       NULL,
                       DT_FP32,
                       dtype,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        grad_output_fp32_local_addr = grad_output_local_addrs[index];
      }
      // Set tile zeros
      if ( ( work_shape.h * work_shape.w ) % tile != 0 )
      {
        tpu_bdc_set_C ( grad_output_fp32_local_addr + work_shape.h * work_shape.w * sizeof ( float ),
                        zero,
                        &tail_shape,
                        &tile_fp32_stride,
                        DT_FP32 );
      }
      // [ work_shape.n, work_shape.c, DIV_UP ( work_shape.w, tile ), tile ] -> [ work_shape.n, work_shape.c, 1, tile ]
      tpu_bdc_fp_avg_pool2d ( grad_output_reduce_tile_fp32_local_addr,
                              grad_output_fp32_local_addr,
                              &tile_shape,
                              &reduce_tile_kernel,
                              &zero_pad,
                              &stride_one,
                              &dilation_one,
                              DT_FP32,
                              one_fp32 );
    }
    // Grad Weight
    if ( grad_weight_enable || grad_input_enable )
    {
      // Grad Output × Input Normalized
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_cast ( grad_output_input_normalized_fp32_local_addr,
                       grad_output_local_addrs[index],
                       &tile_shape,
                       NULL,
                       NULL,
                       DT_FP32,
                       dtype,
                       RM_HALF_TO_EVEN );
        tpu_bdc_fp_mul ( grad_output_input_normalized_fp32_local_addr,
                         grad_output_input_normalized_fp32_local_addr,
                         input_normalized_fp32_local_addr,
                         &tile_shape,
                         NULL,
                         NULL,
                         NULL,
                         DT_FP32 );
      }
      else
      {
        tpu_bdc_fp_mul ( grad_output_input_normalized_fp32_local_addr,
                         grad_output_local_addrs[index],
                         input_normalized_fp32_local_addr,
                         &tile_shape,
                         NULL,
                         NULL,
                         NULL,
                         DT_FP32 );
      }
      // Set tile zeros
      if ( ( work_shape.h * work_shape.w ) % tile != 0 )
      {
        tpu_bdc_set_C ( grad_output_input_normalized_fp32_local_addr + work_shape.h * work_shape.w * sizeof ( float ),
                        zero,
                        &tail_shape,
                        &tile_fp32_stride,
                        DT_FP32 );
      }
      // [ work_shape.n, work_shape.c, DIV_UP ( work_shape.w, tile ), tile ] -> [ work_shape.n, work_shape.c, 1, tile ]
      tpu_bdc_fp_avg_pool2d ( grad_output_input_normalized_reduce_tile_fp32_local_addr,
                              grad_output_input_normalized_fp32_local_addr,
                              &tile_shape,
                              &reduce_tile_kernel,
                              &zero_pad,
                              &stride_one,
                              &dilation_one,
                              DT_FP32,
                              one_fp32 );
    }
    // Grad Bias
    if ( grad_input_enable || grad_bias_enable )
    {
      if ( dtype != DT_FP32 )
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_bias_fp32_local_addr,
                                grad_output_reduce_tile_fp32_local_addr,
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        tpu_bdc_cast ( grad_bias_local_addrs[index],
                       grad_bias_fp32_local_addr,
                       &reduce_shape,
                       NULL,
                       NULL,
                       dtype,
                       DT_FP32,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_bias_local_addrs[index],
                                grad_output_reduce_tile_fp32_local_addr,
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        grad_bias_fp32_local_addr = grad_bias_local_addrs[index];
      }
      if ( grad_bias_enable )
      {
        l2s_grad_bias = true;
        l2s_grad_bias_local_addr = grad_bias_local_addrs[index];
        l2s_grad_bias_global_addr = grad_bias_global_addr + 1UL * cdone * dsize;
        l2s_grad_bias_shape = channel_shape;
      }
    }
    // Grad Weight
    if ( grad_input_enable || grad_weight_enable )
    {
      if ( dtype != DT_FP32 )
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_weight_fp32_local_addr,
                                grad_output_input_normalized_reduce_tile_fp32_local_addr,
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        tpu_bdc_cast ( grad_weight_local_addrs[index],
                       grad_weight_fp32_local_addr,
                       &reduce_shape,
                       NULL,
                       NULL,
                       dtype,
                       DT_FP32,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_weight_local_addrs[index],
                                grad_output_input_normalized_reduce_tile_fp32_local_addr,
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        grad_weight_fp32_local_addr = grad_weight_local_addrs[index];
      }
      if ( grad_weight_enable )
      {
        l2s_grad_weight = true;
        l2s_grad_weight_local_addr = grad_weight_local_addrs[index];
        l2s_grad_weight_global_addr = grad_weight_global_addr + 1UL * cdone * dsize;
        l2s_grad_weight_shape = channel_shape;
      }
    }
    // Grad input
    if ( grad_input_enable )
    {
      // Input Normalized = - ( Input Normalized × Grad Weight + Grad Bias ) / NHW
      tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                       input_normalized_fp32_local_addr,
                       grad_weight_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
      tpu_bdc_fp_add ( input_normalized_fp32_local_addr,
                       input_normalized_fp32_local_addr,
                       grad_bias_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
      tpu_bdc_fp_mul_C ( input_normalized_fp32_local_addr,
                         input_normalized_fp32_local_addr,
                         neg_inv_cghw_fp32,
                         &tile_shape,
                         NULL,
                         NULL,
                         DT_FP32 );
      // Grad Output = Grad Output + Input Normalized
      tpu_bdc_fp_add ( grad_output_fp32_local_addr,
                       grad_output_fp32_local_addr,
                       input_normalized_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       NULL,
                       DT_FP32 );
      // Grad Input = Saved Invstd × Grad Output
      tpu_bdc_fp_mul ( grad_input_local_addrs[index],
                       grad_output_fp32_local_addr,
                       saved_invstd_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       &channel_stride,
                       DT_FP32 );
      l2s_grad_input = true;
      l2s_grad_input_local_addr = grad_input_local_addrs[index];
      l2s_grad_input_global_addr = grad_input_global_addr + 1UL * cdone * grad_input_global_stride.c * dsize;
      l2s_grad_input_shape = work_shape;
    }
    ctodo -= work_shape.c;
    cdone += work_shape.c;
    index = 1 - index;
  }
  
  // Synchronize Point
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  // Move Grad Bias from local memory to global memory
  if ( l2s_grad_bias )
  {
    tpu_gdma_cpy_L2S ( l2s_grad_bias_global_addr,
                       l2s_grad_bias_local_addr,
                       &l2s_grad_bias_shape,
                       NULL,
                       NULL,
                       dtype );
    l2s_grad_bias = false;
  }
  // Move Grad Weight from local memory to global memory
  if ( l2s_grad_weight )
  {
    tpu_gdma_cpy_L2S ( l2s_grad_weight_global_addr,
                       l2s_grad_weight_local_addr,
                       &l2s_grad_weight_shape,
                       NULL,
                       NULL,
                       dtype );
    l2s_grad_weight = false;
  }
  // Move Grad Input from local memory to global memory
  if ( l2s_grad_input )
  {
    tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                       l2s_grad_input_local_addr,
                       &l2s_grad_input_shape,
                       NULL,
                       NULL,
                       dtype );
    l2s_grad_input = false;
  }
  // In group_norm, need to reshape grad_input to (N, C, H, W) for weight's shape is (C,)
  if (group_nums != 0) {
    tpu_continuous_stride ( &grad_input_global_stride, &shape );
    local_addr_t buffer_grad_input_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
    cmax = shape.c;
    while (true) {
      int channel_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype ); next = 0;
      buffer_grad_input_local_addrs[0] = next; next += shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( shape.h * shape.w, tile ), tile, dtype );
      buffer_grad_input_local_addrs[1] = next; next += shape.n * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( shape.h * shape.w, tile ), tile, dtype );
      weight_local_addrs[0] = next; next += channel_size;
      weight_local_addrs[1] = next; next += channel_size;
      weight_fp32_local_addr = next; next += channel_size;
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
    ctodo = shape.c; cdone = 0; index = 0;
    work_shape = shape;
    while (ctodo != 0)
    {
      work_shape.c = MIN( ctodo, cmax );
      dim4 tile_shape = { .n = work_shape.n, .c = work_shape.c, .h = DIV_UP ( ( work_shape.h * work_shape.w ), tile ), .w = tile };
      dim4 channel_shape = { .n = 1, .c = work_shape.c, .h = 1, .w = 1 };
      dim4 channel_stride; tpu_aligned_stride ( &channel_stride, 0, &channel_shape, dtype );
      channel_stride.n = 0, channel_stride.h = 0, channel_stride.w = 0;
      if ( grad_input_enable )
      {
        tpu_gdma_cpy_S2L (  weight_local_addrs[index],
                            weight_global_addr + 1UL * cdone * dsize,
                            &channel_shape,
                            NULL,
                            NULL,
                            dtype );
        if ( l2s_grad_input )
        {
          tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                            l2s_grad_input_local_addr,
                            &l2s_grad_input_shape,
                            &grad_input_global_stride,
                            NULL,
                            dtype );
          l2s_grad_input = false;
        }
        if ( grad_input_enable && dtype != DT_FP32 )
        {
          tpu_bdc_cast ( weight_fp32_local_addr,
                        weight_local_addrs[index],
                        &channel_shape,
                        NULL,
                        NULL,
                        DT_FP32,
                        dtype,
                        RM_HALF_TO_EVEN );
        } else {
          weight_fp32_local_addr = weight_local_addrs[index];
        }
        // reshape grad_input to (shape.N ,shape.C, shape.H, shape.W) in local.
        tpu_gdma_cpy_S2L ( buffer_grad_input_local_addrs[index],
                          grad_input_global_addr + 1UL * cdone * grad_input_global_stride.c * dsize,
                          &work_shape,
                          NULL,
                          NULL,
                          dtype );
        // Synchronize Point
        if ( tpu_is_parallel_state() )
        {
          tpu_parallel_end();
        }
        tpu_parallel_start();
        // grad_input = weight * grad_output
        tpu_bdc_fp_mul( buffer_grad_input_local_addrs[index],
                        buffer_grad_input_local_addrs[index],
                        weight_fp32_local_addr,
                        &tile_shape,
                        NULL,
                        NULL,
                        &channel_stride,
                        DT_FP32 );
        l2s_grad_input = true;
        l2s_grad_input_local_addr = buffer_grad_input_local_addrs[index];
        l2s_grad_input_global_addr = grad_input_global_addr + 1UL * cdone * grad_input_global_stride.c * dsize;
        l2s_grad_input_shape = work_shape;
      }
      ctodo -= work_shape.c;
      cdone += work_shape.c;
      index = 1 - index;
    }
  }
  // Synchronize Point
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s_grad_input )
  {
    tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                       l2s_grad_input_local_addr,
                       &l2s_grad_input_shape,
                       NULL,
                       NULL,
                       dtype );
    l2s_grad_input = false;
  }

  return true;
}

static inline void nodechip_groupnorm2d_backward_all_fp32 (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t saved_mean_global_addr,
global_addr_t saved_invstd_global_addr,
global_addr_t grad_input_global_addr,
global_addr_t grad_weight_global_addr,
global_addr_t grad_bias_global_addr,
dim4 shape,
data_type_t dtype,
int grad_input_enable,
int grad_weight_enable,
int grad_bias_enable,
int group_nums )
{
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  dim4 grouped_shape = {.n = shape.n, .c = group_nums, .h = (shape.c / group_nums) * shape.h, .w = shape.w};
  dim4 input_global_stride; tpu_continuous_stride ( &input_global_stride, &grouped_shape );
  dim4 grad_output_global_stride; tpu_continuous_stride ( &grad_output_global_stride, &grouped_shape );
  dim4 grad_input_global_stride; tpu_continuous_stride ( &grad_input_global_stride, &grouped_shape );
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  const scalar_t neg_inv_cghw_fp32 = { .f32 = -1.0 / ( 1UL * grouped_shape.h * grouped_shape.w ) };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  local_addr_t input_local_addrs[2] = { DEFAULT_LOCAL_ADDR }, grad_output_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t input_normalized_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t saved_mean_local_addr = DEFAULT_LOCAL_ADDR, saved_invstd_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t saved_mean_fp32_local_addr = DEFAULT_LOCAL_ADDR, saved_invstd_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_output_reduce_tile_fp32_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t grad_output_reduce_tile_n_piece_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_bias_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_output_input_normalized_reduce_tile_fp32_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t grad_output_input_normalized_reduce_tile_n_piece_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_weight_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t common_tile_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_bias_fp32_local_addr = DEFAULT_LOCAL_ADDR, grad_weight_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t grad_input_local_addrs[2] = { DEFAULT_LOCAL_ADDR }, weight_local_addr = DEFAULT_LOCAL_ADDR, weight_fp32_local_addr = DEFAULT_LOCAL_ADDR;
  local_addr_t buffer_grad_input_local_addrs[2] = { DEFAULT_LOCAL_ADDR };
  local_addr_t next = 0;
  int nmax = grouped_shape.n, hwmax = grouped_shape.h * grouped_shape.w, cmax = grouped_shape.c, cg = shape.c / group_nums;
  while ( true )
  {
    int tile_size = nmax * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( hwmax, tile ), tile, dtype );
    int tile_fp32_size = nmax * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( hwmax, tile ), tile, DT_FP32 );
    int nc_size = nmax * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    int nc_fp32_size = nmax * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    int reduce_tile_fp32_size = nmax * DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    int reduce_tile_n_piece_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    int reduce_fp32_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    int grad_bias_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    int grad_weight_size = DIV_UP ( cmax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    next = 0;
    input_local_addrs[0] = next; next += tile_size;
    input_local_addrs[1] = next; next += tile_size;
    grad_output_local_addrs[0] = next; next += tile_size;
    grad_output_local_addrs[1] = next; next += tile_size;
    saved_mean_local_addr = next; next += nc_size;
    if ( dtype != DT_FP32 )
    {
      saved_mean_fp32_local_addr = next; next += nc_fp32_size;
    }
    saved_invstd_local_addr = next; next += nc_size;
    if ( dtype != DT_FP32 )
    {
      saved_invstd_fp32_local_addr = next; next += nc_fp32_size;
    }
    if ( grad_input_enable )
    {
      weight_local_addr = next; next += DIV_UP ( cmax * cg, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
      weight_fp32_local_addr = next; next += DIV_UP ( cmax * cg, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
      buffer_grad_input_local_addrs[0] = next; next += nmax * DIV_UP ( cmax * cg, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( hwmax, tile ), tile, dtype );
      buffer_grad_input_local_addrs[1] = next; next += nmax * DIV_UP ( cmax * cg, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( hwmax, tile ), tile, dtype );
      grad_input_local_addrs[0] = next; next += tile_size;
      grad_input_local_addrs[1] = next; next += tile_size;
    }
    if ( grad_input_enable || grad_bias_enable )
    {
      grad_output_reduce_tile_fp32_local_addrs[0] = next; next += reduce_tile_fp32_size;
      grad_output_reduce_tile_fp32_local_addrs[1] = next; next += reduce_tile_fp32_size;
      grad_output_reduce_tile_n_piece_fp32_local_addr = next; next += reduce_tile_n_piece_fp32_size;
      if ( dtype != DT_FP32 )
      {
        grad_bias_fp32_local_addr = next; next += reduce_fp32_size;
      }
      grad_bias_local_addr = next; next += grad_bias_size;
    }
    if ( grad_weight_enable || grad_input_enable )
    {
      grad_output_input_normalized_reduce_tile_fp32_local_addrs[0] = next; next += reduce_tile_fp32_size;
      grad_output_input_normalized_reduce_tile_fp32_local_addrs[1] = next; next += reduce_tile_fp32_size;
      grad_output_input_normalized_reduce_tile_n_piece_fp32_local_addr = next; next += reduce_tile_n_piece_fp32_size;
      if ( dtype != DT_FP32 )
      {
        grad_weight_fp32_local_addr = next; next += reduce_fp32_size;
      }
      grad_weight_local_addr = next; next += grad_weight_size;
    }
    if ( dtype != DT_FP32 )
    {
      input_normalized_fp32_local_addr = next; next += tile_fp32_size;
      common_tile_fp32_local_addr = next; next += tile_fp32_size;
    }
    if ( ( int ) next > LOCAL_MEM_SIZE )
    {
      if ( cmax > NPU_NUM )
      {
        cmax -= ( ( cmax % NPU_NUM > 0 ) ? ( cmax % NPU_NUM ) : NPU_NUM );
        continue;
      }
      else if ( nmax > 1 )
      {
        nmax /= 2;
      }
      else if ( hwmax > 1 )
      {
        hwmax /= 2;
      }
      else
      {
        TPUKERNEL_ASSERT ( 0 );
      }
    }
    else
    {
      break;
    }
  }
  bool l2s_grad_input = false;
  local_addr_t l2s_grad_input_local_addr = DEFAULT_LOCAL_ADDR;
  global_addr_t l2s_grad_input_global_addr = DEFAULT_LOCAL_ADDR;
  dim4 l2s_grad_input_shape;
  dim4 work_shape; work_shape.n = grouped_shape.n;
  int ctodo = grouped_shape.c, cdone = 0;
  while ( ctodo != 0 )
  {
    work_shape.c = MIN ( ctodo, cmax );
    dim4 channel_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = 1 };
    dim4 channel_stride; tpu_aligned_stride ( &channel_stride, 0, &channel_shape, dtype );
    channel_stride.h = 0, channel_stride.w = 0;
    dim4 weight_shape = { .n = 1, .c = work_shape.c * cg, .h = 1, .w = 1 };
    dim4 weight_stride; tpu_aligned_stride ( &weight_stride, 0, &weight_shape, dtype );
    weight_stride.n = 0, weight_stride.h = 0, weight_stride.w = 0;
    dim4 reduce_tile_n_piece_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = tile };
    dim2 reduce_kernel = { .h = 1, .w = tile };
    dim4 reduce_shape = { .n = 1, .c = work_shape.c, .h = 1, .w = 1 };
    // Move Saved Mean from global to local memory
    tpu_gdma_cpy_S2L ( saved_mean_local_addr,
                       saved_mean_global_addr + 1UL * cdone * dsize,
                       &channel_shape,
                       NULL,
                       NULL,
                       dtype );
    // Move Saved Invstd from global to lobal memory
    tpu_gdma_cpy_S2L ( saved_invstd_local_addr,
                       saved_invstd_global_addr + 1UL * cdone * dsize,
                       &channel_shape,
                       NULL,
                       NULL,
                       dtype );
    tpu_gdma_cpy_S2L ( weight_local_addr,
                         weight_global_addr + 1UL * cdone * dsize,
                         &weight_shape,
                         NULL,
                         NULL,
                         dtype );
    // Set ∑<n ( partial ), h, w> Grad Output zeros
    if ( grad_input_enable || grad_bias_enable )
    {
      tpu_bdc_set_C ( grad_output_reduce_tile_n_piece_fp32_local_addr,
                      zero,
                      &reduce_tile_n_piece_shape,
                      NULL,
                      DT_FP32 );
    }
    // Set ∑<n ( partial ), h, w> Grad Output x Intpu Normalized zeros
    if ( grad_input_enable || grad_weight_enable )
    {
      tpu_bdc_set_C ( grad_output_input_normalized_reduce_tile_n_piece_fp32_local_addr,
                      zero,
                      &reduce_tile_n_piece_shape,
                      NULL,
                      DT_FP32 );
    }
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast ( saved_mean_fp32_local_addr,
                     saved_mean_local_addr,
                     &channel_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
      tpu_bdc_cast ( saved_invstd_fp32_local_addr,
                     saved_invstd_local_addr,
                     &channel_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
      if ( grad_input_enable )
      {
        tpu_bdc_cast ( weight_fp32_local_addr,
                       weight_local_addr,
                       &weight_shape,
                       NULL,
                       NULL,
                       DT_FP32,
                       dtype,
                       RM_HALF_TO_EVEN );
      }
    }
    else
    {
      saved_mean_fp32_local_addr = saved_mean_local_addr;
      saved_invstd_fp32_local_addr = saved_invstd_local_addr;
      if ( grad_input_enable )
      {
        weight_fp32_local_addr = weight_local_addr;
      }
    }
    int index = 0;
    int ntodo = grouped_shape.n, ndone = 0;
    while ( ntodo != 0 )
    {
      work_shape.n = MIN ( ntodo, nmax );
      dim4 reduce_tile_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = tile };
      dim4 reduce_tile_fp32_stride; tpu_aligned_stride ( &reduce_tile_fp32_stride, 0, &reduce_tile_shape, DT_FP32 );
      int hwtodo = grouped_shape.h * grouped_shape.w, hwdone = 0;
      while ( hwtodo != 0 )
      {
        work_shape.h = 1;
        work_shape.w = MIN ( hwtodo, hwmax );
        // Move Input from global memory to local memory
        tpu_gdma_cpy_S2L ( input_local_addrs[index],
                           input_global_addr + (
                           1UL * ndone * input_global_stride.n +
                           1UL * cdone * input_global_stride.c +
                           1UL * hwdone * input_global_stride.w ) * dsize,
                           &work_shape,
                           NULL, // tile is eu num of dtype, so NULL is OK
                           &input_global_stride,
                           dtype );
        // Move Grad Output from global memory to local memory
        tpu_gdma_cpy_S2L ( grad_output_local_addrs[index],
                           grad_output_global_addr + (
                           1UL * ndone * grad_output_global_stride.n +
                           1UL * cdone * grad_output_global_stride.c +
                           1UL * hwdone * grad_output_global_stride.w ) * dsize,
                           &work_shape,
                           NULL, // tile is eu num of dtype, so NULL is OK
                           &grad_output_global_stride,
                           dtype );
        if ( tpu_is_parallel_state() )
        {
          tpu_parallel_end();
        }
        tpu_parallel_start();
        dim4 tile_shape = { .n = work_shape.n, .c = work_shape.c, .h = DIV_UP ( ( work_shape.h * work_shape.w ), tile ), .w = tile };
        dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
        dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
        dim4 tail_shape = { .n = work_shape.n, .c = work_shape.c, .h = 1, .w = tile - ( ( work_shape.h * work_shape.w ) % tile ) };
        dim2 reduce_tile_kernel = { .h = tile_shape.h, .w = 1 };
        // Grad Bias
        if ( grad_input_enable || grad_bias_enable )
        {
          local_addr_t grad_output_fp32_local_addr = DEFAULT_LOCAL_ADDR;
          if ( dtype != DT_FP32 )
          {
            grad_output_fp32_local_addr = common_tile_fp32_local_addr;
            tpu_bdc_cast ( grad_output_fp32_local_addr,
                           grad_output_local_addrs[index],
                           &tile_shape,
                           NULL,
                           NULL,
                           DT_FP32,
                           dtype,
                           RM_HALF_TO_EVEN );
          }
          else
          {
            grad_output_fp32_local_addr = grad_output_local_addrs[index];
          }
          // Set tile zeros
          if ( ( work_shape.h * work_shape.w ) % tile != 0 )
          {
            tpu_bdc_set_C ( grad_output_fp32_local_addr + work_shape.h * work_shape.w * sizeof ( float ),
                            zero,
                            &tail_shape,
                            &tile_fp32_stride,
                            DT_FP32 );
          }
          // [ work_shape.n, work_shape.c, DIV_UP ( work_shape.w, tile ), tile ] -> [ work_shape.n, work_shape.c, 1, tile ]
          tpu_bdc_fp_avg_pool2d ( grad_output_reduce_tile_fp32_local_addrs[index],
                                  grad_output_fp32_local_addr,
                                  &tile_shape,
                                  &reduce_tile_kernel,
                                  &zero_pad,
                                  &stride_one,
                                  &dilation_one,
                                  DT_FP32,
                                  one_fp32 );
          if ( hwdone > 0 )
          {
            tpu_bdc_fp_add ( grad_output_reduce_tile_fp32_local_addrs[index],
                             grad_output_reduce_tile_fp32_local_addrs[index],
                             grad_output_reduce_tile_fp32_local_addrs[1 - index],
                             &reduce_tile_shape,
                             NULL,
                             NULL,
                             NULL,
                             DT_FP32 );
          }
        }
        // Input Normalized = ( Input − Saved Mean ) × Saved Invstd
        if ( dtype != DT_FP32 )
        {
          tpu_bdc_cast ( input_normalized_fp32_local_addr,
                         input_local_addrs[index],
                         &tile_shape,
                         NULL,
                         NULL,
                         DT_FP32,
                         dtype,
                         RM_HALF_TO_EVEN );
          tpu_bdc_fp_sub ( input_normalized_fp32_local_addr,
                           input_normalized_fp32_local_addr,
                           saved_mean_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
          tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                           input_normalized_fp32_local_addr,
                           saved_invstd_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
        }
        else
        {
          input_normalized_fp32_local_addr = input_local_addrs[index];
          tpu_bdc_fp_sub ( input_normalized_fp32_local_addr,
                           input_local_addrs[index],
                           saved_mean_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
          tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                           input_normalized_fp32_local_addr,
                           saved_invstd_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
        }
        // Grad Weight
        if ( grad_weight_enable || grad_input_enable )
        {
          // Grad Output × Input Normalized
          local_addr_t grad_output_input_normalized_fp32_local_addr = DEFAULT_LOCAL_ADDR;
          if ( dtype != DT_FP32 )
          {
            grad_output_input_normalized_fp32_local_addr = common_tile_fp32_local_addr;
            tpu_bdc_cast ( grad_output_input_normalized_fp32_local_addr,
                           grad_output_local_addrs[index],
                           &tile_shape,
                           NULL,
                           NULL,
                           DT_FP32,
                           dtype,
                           RM_HALF_TO_EVEN );
            tpu_bdc_fp_mul ( grad_output_input_normalized_fp32_local_addr,
                             grad_output_input_normalized_fp32_local_addr,
                             input_normalized_fp32_local_addr,
                             &tile_shape,
                             NULL,
                             NULL,
                             NULL,
                             DT_FP32 );
          }
          else
          {
            grad_output_input_normalized_fp32_local_addr = grad_output_local_addrs[index];
            tpu_bdc_fp_mul ( grad_output_input_normalized_fp32_local_addr,
                             grad_output_local_addrs[index],
                             input_normalized_fp32_local_addr,
                             &tile_shape,
                             NULL,
                             NULL,
                             NULL,
                             DT_FP32 );
          }
          // Set tile zeros
          if ( ( work_shape.h * work_shape.w ) % tile != 0 )
          {
            tpu_bdc_set_C ( grad_output_input_normalized_fp32_local_addr + work_shape.h * work_shape.w * sizeof ( float ),
                            zero,
                            &tail_shape,
                            &tile_fp32_stride,
                            DT_FP32 );
          }
          // [ work_shape.n, work_shape.c, DIV_UP ( work_shape.w, tile ), tile ] -> [ work_shape.n, work_shape.c, 1, tile ]
          tpu_bdc_fp_avg_pool2d ( grad_output_input_normalized_reduce_tile_fp32_local_addrs[index],
                                  grad_output_input_normalized_fp32_local_addr,
                                  &tile_shape,
                                  &reduce_tile_kernel,
                                  &zero_pad,
                                  &stride_one,
                                  &dilation_one,
                                  DT_FP32,
                                  one_fp32 );
          if ( hwdone > 0 )
          {
            tpu_bdc_fp_add ( grad_output_input_normalized_reduce_tile_fp32_local_addrs[index],
                             grad_output_input_normalized_reduce_tile_fp32_local_addrs[index],
                             grad_output_input_normalized_reduce_tile_fp32_local_addrs[1 - index],
                             &reduce_tile_shape,
                             NULL,
                             NULL,
                             NULL,
                             DT_FP32 );
          }
        }
        hwtodo -= work_shape.h * work_shape.w;
        hwdone += work_shape.h * work_shape.w;
        index = 1 - index;
      }
      ntodo -= work_shape.n;
      ndone += work_shape.n;
    }
    // Grad Bias
    if ( grad_input_enable || grad_bias_enable )
    {
      if ( dtype != DT_FP32 )
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_bias_fp32_local_addr,
                                grad_output_reduce_tile_fp32_local_addrs[1 - index],
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        tpu_bdc_cast ( grad_bias_local_addr,
                       grad_bias_fp32_local_addr,
                       &reduce_shape,
                       NULL,
                       NULL,
                       dtype,
                       DT_FP32,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_bias_local_addr,
                                grad_output_reduce_tile_fp32_local_addrs[1 - index],
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        grad_bias_fp32_local_addr = grad_bias_local_addr;
      }
    }
    // Grad Weight
    if ( grad_input_enable || grad_weight_enable )
    {
      if ( dtype != DT_FP32 )
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_weight_fp32_local_addr,
                                grad_output_input_normalized_reduce_tile_fp32_local_addrs[1 - index],
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        tpu_bdc_cast ( grad_weight_local_addr,
                       grad_weight_fp32_local_addr,
                       &reduce_shape,
                       NULL,
                       NULL,
                       dtype,
                       DT_FP32,
                       RM_HALF_TO_EVEN );
      }
      else
      {
        // [ 1, work_shape.c, 1, tile ] -> [ 1, work_shape.c, 1, 1 ]
        tpu_bdc_fp_avg_pool2d ( grad_weight_local_addr,
                                grad_output_input_normalized_reduce_tile_fp32_local_addrs[1 - index],
                                &reduce_tile_n_piece_shape,
                                &reduce_kernel,
                                &zero_pad,
                                &stride_one,
                                &dilation_one,
                                DT_FP32,
                                one_fp32 );
        grad_weight_fp32_local_addr = grad_weight_local_addr;
      }
    }
    // Synchronize Point
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    if ( grad_bias_enable )
    {
      tpu_gdma_cpy_L2S ( grad_bias_global_addr + 1UL * cdone * dsize,
                         grad_bias_local_addr,
                         &channel_shape,
                         NULL,
                         NULL,
                         dtype );
    }
    if ( grad_weight_enable )
    {
      tpu_gdma_cpy_L2S ( grad_weight_global_addr + 1UL * cdone * dsize,
                         grad_weight_local_addr,
                         &channel_shape,
                         NULL,
                         NULL,
                         dtype );
    }
    // Grad input
    if ( grad_input_enable )
    {
      int index = 0;
      int ntodo = grouped_shape.n, ndone = 0;
      while ( ntodo != 0 )
      {
        work_shape.n = MIN ( ntodo, nmax );
        int hwtodo = grouped_shape.h * grouped_shape.w, hwdone = 0;
        while ( hwtodo != 0 )
        {
          work_shape.h = 1;
          work_shape.w = MIN ( hwtodo, hwmax );
          // Move Input from global memory to local memory
          tpu_gdma_cpy_S2L ( input_local_addrs[index],
                             input_global_addr + (
                             1UL * ndone * input_global_stride.n +
                             1UL * cdone * input_global_stride.c +
                             1UL * hwdone * input_global_stride.w ) * dsize,
                             &work_shape,
                             NULL, // tile is eu num of dtype, so NULL is OK
                             &input_global_stride,
                             dtype );
          // Move Grad Output from global memory to local memory
          tpu_gdma_cpy_S2L ( grad_output_local_addrs[index],
                             grad_output_global_addr + (
                             1UL * ndone * grad_output_global_stride.n +
                             1UL * cdone * grad_output_global_stride.c +
                             1UL * hwdone * grad_output_global_stride.w ) * dsize,
                             &work_shape,
                             NULL, // tile is eu num of dtype, so NULL is OK
                             &grad_output_global_stride,
                             dtype );
          if ( tpu_is_parallel_state() )
          {
            tpu_parallel_end();
          }
          tpu_parallel_start();
          // Move Grad Input from local memory to global memory
          if ( l2s_grad_input )
          {
            tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                               l2s_grad_input_local_addr,
                               &l2s_grad_input_shape,
                               &grad_input_global_stride,
                               NULL,
                               dtype );
            dim4 buffer_shape = { .n = l2s_grad_input_shape.n, .c = l2s_grad_input_shape.c*cg, .h = l2s_grad_input_shape.h, .w = l2s_grad_input_shape.w/cg };
            tpu_gdma_cpy_S2L ( buffer_grad_input_local_addrs[index],
                               l2s_grad_input_global_addr,
                               &buffer_shape,
                               NULL,
                               &grad_input_global_stride,
                               dtype);
            tpu_bdc_fp_mul ( buffer_grad_input_local_addrs[index],
                             buffer_grad_input_local_addrs[index],
                             weight_fp32_local_addr,
                             &weight_shape,
                             NULL,
                             NULL,
                             &weight_stride,
                             DT_FP32 );
            tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                               buffer_grad_input_local_addrs[index],
                               &buffer_shape,
                               &grad_input_global_stride,
                               NULL,
                               dtype );
            l2s_grad_input = false;
          }
          dim4 tile_shape = { .n = work_shape.n, .c = work_shape.c, .h = DIV_UP ( ( work_shape.h * work_shape.w ), tile ), .w = tile };
          dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
          dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
          // Input Normalized = ( Input − Saved Mean ) × Saved Invstd
          if ( dtype != DT_FP32 )
          {
            tpu_bdc_cast ( input_normalized_fp32_local_addr,
                           input_local_addrs[index],
                           &tile_shape,
                           NULL,
                           NULL,
                           DT_FP32,
                           dtype,
                           RM_HALF_TO_EVEN );
            tpu_bdc_fp_sub ( input_normalized_fp32_local_addr,
                             input_normalized_fp32_local_addr,
                             saved_mean_fp32_local_addr,
                             &tile_shape,
                             NULL,
                             NULL,
                             &channel_stride,
                             DT_FP32 );
            tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                             input_normalized_fp32_local_addr,
                             saved_invstd_fp32_local_addr,
                             &tile_shape,
                             NULL,
                             NULL,
                             &channel_stride,
                             DT_FP32 );
          }
          else
          {
            input_normalized_fp32_local_addr = input_local_addrs[index];
            tpu_bdc_fp_sub ( input_normalized_fp32_local_addr,
                             input_local_addrs[index],
                             saved_mean_fp32_local_addr,
                             &tile_shape,
                             NULL,
                             NULL,
                             &channel_stride,
                             DT_FP32 );
            tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                             input_normalized_fp32_local_addr,
                             saved_invstd_fp32_local_addr,
                             &tile_shape,
                             NULL,
                             NULL,
                             &channel_stride,
                             DT_FP32 );
          }
          // Input Normalized = - ( Input Normalized × Grad Weight + Grad Bias ) / cgHW
          tpu_bdc_fp_mul ( input_normalized_fp32_local_addr,
                           input_normalized_fp32_local_addr,
                           grad_weight_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
          tpu_bdc_fp_add ( input_normalized_fp32_local_addr,
                           input_normalized_fp32_local_addr,
                           grad_bias_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
          tpu_bdc_fp_mul_C ( input_normalized_fp32_local_addr,
                             input_normalized_fp32_local_addr,
                             neg_inv_cghw_fp32,
                             &tile_shape,
                             NULL,
                             NULL,
                             DT_FP32 );
          local_addr_t grad_output_fp32_local_addr = DEFAULT_LOCAL_ADDR;
          if ( dtype != DT_FP32 )
          {
            grad_output_fp32_local_addr = common_tile_fp32_local_addr;
            tpu_bdc_cast ( grad_output_fp32_local_addr,
                           grad_output_local_addrs[index],
                           &tile_shape,
                           NULL,
                           NULL,
                           DT_FP32,
                           dtype,
                           RM_HALF_TO_EVEN );
          }
          else
          {
            grad_output_fp32_local_addr = grad_output_local_addrs[index];
          }
          // Grad Output = Grad Output + Input Normalized
          tpu_bdc_fp_add ( grad_output_fp32_local_addr,
                           grad_output_fp32_local_addr,
                           input_normalized_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           NULL,
                           DT_FP32 );
          // Grad Input = Weight × Saved Invstd × Grad Output
          tpu_bdc_fp_mul ( grad_input_local_addrs[index],
                           grad_output_fp32_local_addr,
                           saved_invstd_fp32_local_addr,
                           &tile_shape,
                           NULL,
                           NULL,
                           &channel_stride,
                           DT_FP32 );
          l2s_grad_input = true;
          l2s_grad_input_local_addr = grad_input_local_addrs[index];
          l2s_grad_input_global_addr = grad_input_global_addr + (
                                       1UL * ndone * grad_input_global_stride.n +
                                       1UL * cdone * grad_input_global_stride.c +
                                       1UL * hwdone * grad_input_global_stride.w ) * dsize;
          l2s_grad_input_shape = work_shape;
          hwtodo -= work_shape.h * work_shape.w;
          hwdone += work_shape.h * work_shape.w;
          index = 1 - index;
        }
        ntodo -= work_shape.n;
        ndone += work_shape.n;
      }
      // Synchronize Point
      if ( tpu_is_parallel_state() )
      {
        tpu_parallel_end();
      }
      // Move Grad Input from local memory to global memory
      if ( l2s_grad_input )
      {
        tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                           l2s_grad_input_local_addr,
                           &l2s_grad_input_shape,
                           &grad_input_global_stride,
                           NULL,
                           dtype );
        dim4 buffer_shape = { .n = l2s_grad_input_shape.n, .c = l2s_grad_input_shape.c*cg, .h = l2s_grad_input_shape.h, .w = l2s_grad_input_shape.w/cg };
        tpu_gdma_cpy_S2L ( buffer_grad_input_local_addrs[index],
                            l2s_grad_input_global_addr,
                            &buffer_shape,
                            NULL,
                            &grad_input_global_stride,
                            dtype);
        tpu_bdc_fp_mul ( buffer_grad_input_local_addrs[index],
                          buffer_grad_input_local_addrs[index],
                          weight_fp32_local_addr,
                          &weight_shape,
                          NULL,
                          NULL,
                          &weight_stride,
                          DT_FP32 );
        tpu_gdma_cpy_L2S ( l2s_grad_input_global_addr,
                            buffer_grad_input_local_addrs[index],
                            &buffer_shape,
                            &grad_input_global_stride,
                            NULL,
                            dtype );
        l2s_grad_input = false;
      }
    }
    ctodo -= work_shape.c;
    cdone += work_shape.c;
  }

}


void tpu_kernel_api_groupnorm2d_backward ( const void *args )
{
  sg_api_groupnorm2d_backward_t * api = ( sg_api_groupnorm2d_backward_t * ) args;
  dim4 shape = { .n = api->shape[0], .c = api->shape[1], .h = api->shape[2], .w = api->shape[3] };
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  if ( api->grad_input_global_addr != 0 || api->grad_weight_global_addr != 0 )
  {
    bool split_c_only = nodechip_groupnorm2d_backward_split_c_only_all_fp32 (
                        api->grad_output_global_addr,
                        api->input_global_addr,
                        api->weight_global_addr,
                        api->saved_mean_global_addr,
                        api->saved_invstd_global_addr,
                        api->grad_input_global_addr,
                        api->grad_weight_global_addr,
                        api->grad_bias_global_addr,
                        shape,
                        ( data_type_t ) api->dtype,
                        api->grad_input_global_addr != 0,
                        api->grad_weight_global_addr != 0,
                        api->grad_bias_global_addr != 0,
                        api->group_nums );
    if ( !split_c_only )
    {
      nodechip_groupnorm2d_backward_all_fp32 (
      api->grad_output_global_addr,
      api->input_global_addr,
      api->weight_global_addr,
      api->saved_mean_global_addr,
      api->saved_invstd_global_addr,
      api->grad_input_global_addr,
      api->grad_weight_global_addr,
      api->grad_bias_global_addr,
      shape,
      ( data_type_t ) api->dtype,
      api->grad_input_global_addr != 0,
      api->grad_weight_global_addr != 0,
      api->grad_bias_global_addr != 0,
      api->group_nums );
    }
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_groupnorm2d_backward );
