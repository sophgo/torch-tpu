#include "sg_api_struct.h"
#include "tpu_kernel.h"


uint64_t norm_tpu_aligned_feature_size(int h, int w, data_type_t dtype) {
    uint64_t align_h_w = ALIGN(h * w, tpu_eu_num(dtype));
    return ALIGN(align_h_w * (uint64_t) tpu_data_type_bits(dtype), 8) / 8;
}

void nodechip_norm2 (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int len,
data_type_t dtype,
bool do_sqrt )
{
  const int dsize = tpu_data_type_size ( dtype );
  const int tile = tpu_eu_num ( dtype );
  const scalar_t zero = { .u32 = 0 };
  const scalar_t one_fp32 = { .f32 = 1.f };
  const dim2 stride_one = { .h = 1, .w = 1 };
  const dim2 dilation_one = { .h = 1, .w = 1 };
  const padding_t zero_pad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  local_addr_t input_local_addrs[2];
  local_addr_t input_fp32_local_addr;
  local_addr_t reduce_tile_fp32_local_addrs[2];
  local_addr_t reduce_fp32_local_addr;
  local_addr_t reduce_cw_trans_fp32_local_addr;
  local_addr_t output_local_addr;
  int wmax = len;
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    uint64_t input_size = norm_tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, dtype );
    uint64_t input_size_fp32 = norm_tpu_aligned_feature_size ( DIV_UP ( wmax, tile ), tile, DT_FP32 );
    uint64_t reduce_tile_size_fp32 = norm_tpu_aligned_feature_size ( 1, tile, DT_FP32 );
    uint64_t reduce_size_fp32 = norm_tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    uint64_t reduce_cw_trans_size_fp32 = norm_tpu_aligned_feature_size ( 1, NPU_NUM, DT_FP32 );
    uint64_t output_fp32_size = norm_tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    input_local_addrs[0] = next; next += input_size;
    input_local_addrs[1] = next; next += input_size;
    input_fp32_local_addr = next; next += input_size_fp32;
    reduce_tile_fp32_local_addrs[0] = next; next += reduce_tile_size_fp32;
    reduce_tile_fp32_local_addrs[1] = next; next += reduce_tile_size_fp32;
    reduce_fp32_local_addr = next; next += reduce_size_fp32;
    reduce_cw_trans_fp32_local_addr = next; next += reduce_cw_trans_size_fp32;
    output_local_addr = next; next += output_fp32_size;
    if ( ( int ) next > LOCAL_MEM_SIZE )
    {
      if ( wmax > 1 )
      {
        wmax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
    else
    {
      break;
    }
  }
  // set reduce_tile_fp32_local_addrs[1] zeros
  dim4 reduce_tile_fp32_full_npu_shape = { .n = 1, .c = NPU_NUM, .h = 1, .w = tile };
  tpu_bdc_set_C ( reduce_tile_fp32_local_addrs[1],
                  zero,
                  &reduce_tile_fp32_full_npu_shape,
                  NULL,
                  DT_FP32 );
  dim4 shape = { .n = 1, .h = 1 };
  int index = 0;
  int todo = len, done = 0;
  while ( todo != 0 )
  {
    if ( todo > NPU_NUM )
    {
      shape.c = NPU_NUM;
      shape.w = MIN ( todo / NPU_NUM, wmax );
    }
    else
    {
      shape.c = todo;
      shape.w = 1;
    }
    dim4 tile_shape = { .n = 1, .c = shape.c, .h = DIV_UP ( shape.w, tile ), .w = tile };
    dim4 tile_stride; tpu_aligned_stride ( &tile_stride, 0, &tile_shape, dtype );
    tpu_gdma_cpy_S2L ( input_local_addrs[index],
                       input_global_addr + done * dsize,
                       &shape,
                       &tile_stride,
                       NULL,
                       dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( dtype == DT_FP32 )
    {
      tpu_bdc_fp_mul ( input_fp32_local_addr,
                       input_local_addrs[index],
                       input_local_addrs[index],
                       &tile_shape,
                       NULL,
                       NULL,
                       NULL,
                       DT_FP32 );
    }
    else
    {
      tpu_bdc_cast ( input_fp32_local_addr,
                     input_local_addrs[index],
                     &tile_shape,
                     NULL,
                     NULL,
                     DT_FP32,
                     dtype,
                     RM_HALF_TO_EVEN );
      tpu_bdc_fp_mul ( input_fp32_local_addr,
                       input_fp32_local_addr,
                       input_fp32_local_addr,
                       &tile_shape,
                       NULL,
                       NULL,
                       NULL,
                       DT_FP32 );
    }
    // set input tail zero
    if ( shape.w % tile != 0 )
    {
      dim4 tile_fp32_stride; tpu_aligned_stride ( &tile_fp32_stride, 0, &tile_shape, DT_FP32 );
      dim4 tail_shape = { .n = 1, .c = shape.c, .h = 1, .w = tile - ( shape.w % tile ) };
      tpu_bdc_set_C ( input_fp32_local_addr + shape.w * sizeof ( float ),
                      zero,
                      &tail_shape,
                      &tile_stride,
                      DT_FP32 );
    }
    // [ 1, shape.c, DIV_UP ( shape.w, tile ), tile ] -> [ 1, shape.c, 1, tile ]
    dim2 kernel = { .h = tile_shape.h, .w = 1 };
    tpu_bdc_fp_avg_pool2d ( reduce_tile_fp32_local_addrs[0],
                            input_fp32_local_addr,
                            &tile_shape,
                            &kernel,
                            &zero_pad,
                            &stride_one,
                            &dilation_one,
                            DT_FP32,
                            one_fp32 );
    dim4 reduce_tile_shape = { .n = 1, .c = shape.c, .h = 1, .w = tile };
    tpu_bdc_fp_add ( reduce_tile_fp32_local_addrs[1],
                     reduce_tile_fp32_local_addrs[1],
                     reduce_tile_fp32_local_addrs[0],
                     &reduce_tile_shape,
                     NULL,
                     NULL,
                     NULL,
                     DT_FP32 );
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  // [ 1, NPU_NUM, 1, tile ] -> [ 1, NPU_NUM, 1, 1 ]
  dim4 reduce_tile_shape = { .n = 1, .c = NPU_NUM, .h = 1, .w = tile };
  dim2 kernel = { .h = 1, .w = tile };
  tpu_bdc_fp_avg_pool2d ( reduce_fp32_local_addr,
                          reduce_tile_fp32_local_addrs[1],
                          &reduce_tile_shape,
                          &kernel,
                          &zero_pad,
                          &stride_one,
                          &dilation_one,
                          DT_FP32,
                          one_fp32 );
  // [ 1, NPU_NUM, 1, 1 ] -> [ 1, 1, 1, NPU_NUM ]
  dim4 cw_trans_shape = { .n = 1, .c = 1, .h = 1, .w = NPU_NUM };
  tpu_bdc_wc_trans ( reduce_cw_trans_fp32_local_addr,
                     reduce_fp32_local_addr,
                     &cw_trans_shape,
                     DT_FP32 );
  // [ 1, 1, 1, NPU_NUM ] -> [ 1, 1, 1, 1 ]
  kernel.h = 1;
  kernel.w = NPU_NUM;
  tpu_bdc_fp_avg_pool2d ( reduce_cw_trans_fp32_local_addr,
                          reduce_cw_trans_fp32_local_addr,
                          &cw_trans_shape,
                          &kernel,
                          &zero_pad,
                          &stride_one,
                          &dilation_one,
                          DT_FP32,
                          one_fp32 );
  dim4 reduce_full_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  if (do_sqrt)
  {  
    tpu_bdc_fp32_sqrt ( output_local_addr,
                        reduce_cw_trans_fp32_local_addr,
                        &reduce_full_shape );
    if ( dtype != DT_FP32 )
    {
      tpu_bdc_cast (output_local_addr,
                    output_local_addr,
                    &reduce_full_shape,
                    NULL,
                    NULL,
                    dtype,
                    DT_FP32,
                    RM_HALF_TO_EVEN );
    }
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  tpu_gdma_cpy_L2S ( output_global_addr,
                     do_sqrt ? output_local_addr : reduce_cw_trans_fp32_local_addr,
                     &reduce_full_shape,
                     NULL,
                     NULL,
                     do_sqrt ? dtype : DT_FP32 );
}

static inline void nodechip_reduce_and_sqrt(
  global_addr_t input_global_addr,
  global_addr_t output_global_addr,
  int len,
  data_type_t dtype
) {
  local_addr_t input_local_addr = 0;
  local_addr_t reduce_local_addr = LOCAL_MEM_SIZE / 2;
  dim4 input_shape = {1, 1, 1, 8};
  dim2 kernel = {1, 8};
  padding_t zero_pad = {0, 0, 0, 0};
  dim2 oneone = {1, 1};
  scalar_t one_fp32 = {.f32 = 1.f };
  tpu_gdma_cpy_S2L(input_local_addr, input_global_addr, &input_shape, NULL, NULL, DT_FP32);
  tpu_bdc_fp_avg_pool2d(reduce_local_addr,
                        input_local_addr,
                        &input_shape,
                        &kernel,
                        &zero_pad,
                        &oneone,
                        &oneone,
                        DT_FP32,
                        one_fp32);
  input_shape.w = 1;
  tpu_bdc_fp32_sqrt(input_local_addr, reduce_local_addr, &input_shape);
  if (dtype != DT_FP32) {
    tpu_bdc_cast(input_local_addr, input_local_addr, &input_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO);
  }
  tpu_gdma_cpy_L2S(output_global_addr, input_local_addr, &input_shape, NULL, NULL, dtype);
}

static inline void nodechip_clear_buffer(
    global_addr_t buffer_global_addr
) {
    dim4 input_shape = {1, 1, 1, 8};
    scalar_t zero_scalar = {.u32 = 0};
    tpu_gdma_set_C_system(buffer_global_addr, zero_scalar, &input_shape, NULL, DT_FP32);
}

int tpu_kernel_api_norm2_multi_core ( const void * args )
{
  sg_api_norm2_multi_core_t * api = ( sg_api_norm2_multi_core_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int len = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    len *= api->shape[i];
  }
  tpu_initialize();
#ifdef ENABLE_MULTI_CORE
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(len, core_num);
  int length_secs = DIV_UP(len, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = len - length_slice * (length_secs - 1);
  // sometimes buffer memory is not empty. clear the buffer
  if (core_idx == 0) {
      nodechip_clear_buffer(api->buffer_global_addr);
  }   
  tpu_sync_all();
  if (core_idx * length_slice < len) {
    nodechip_norm2(api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                   api->buffer_global_addr + core_idx * tpu_data_type_size(DT_FP32),
                   cur_length_slice,
                   (data_type_t)api->dtype, 0 );
  }
  tpu_sync_all();
  // two step calculation. TO DO: use L2mem to reduce
  if (core_idx == 0) {
    nodechip_reduce_and_sqrt(api->buffer_global_addr, api->output_global_addr, len, (data_type_t)api->dtype);
  }
  tpu_poll();
  return 0;
#else
  nodechip_norm2 ( api->input_global_addr, api->output_global_addr, len, ( data_type_t ) api->dtype, 1 );
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_norm2_multi_core );