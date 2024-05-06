#include "sg_api_struct.h"
#include "tpu_kernel.h"


void nodechip_mse_loss_forward (
global_addr_t input_global_addr,
global_addr_t target_global_addr,
global_addr_t output_global_addr,
int length,
int reduction,
data_type_t dtype)
{
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  scalar_t ScaleFP = { .f32 = 1.f };
  const scalar_t zero = { .f32 = 0.f };
  const int dsize = tpu_data_type_size ( DT_FP32 );
  // used in dtype != FP32
  const int dsize_global = tpu_data_type_size ( dtype );
  int wmax = DIV_UP ( length, NPU_NUM );
  dim4 reduce_shape = { .c = 1, .n = 1, .h = 1, .w = 1 };
  local_addr_t input_local_addrs[2], target_local_addrs[2], output_local_addrs[2];
  // used in dtype != FP32
  local_addr_t input_local_addrs_temp = 0, output_local_addrs_temp = 0, reduce_total_local_addrs_temp = 0;
  local_addr_t reduce_split_local_addrs = 0, transpose_reduce_split_local_addrs = 0, reduce_total_local_addrs = 0;
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = tpu_aligned_feature_size ( 1, wmax, DT_FP32 );
    // used in dtype != FP32
    int size_temp = tpu_aligned_feature_size ( 1, wmax, dtype );
    input_local_addrs[0] = next; next += size;
    input_local_addrs[1] = next; next += size;
    target_local_addrs[0] = next; next += size;
    target_local_addrs[1] = next; next += size;
    output_local_addrs[0] = next; next += size;
    output_local_addrs[1] = next; next += size;
    transpose_reduce_split_local_addrs = next; next += NPU_NUM * dsize;
    // used in dtype != FP32
    if ( dtype != DT_FP32 )
    {
      input_local_addrs_temp = next; next += size_temp;
      output_local_addrs_temp = next; next += size_temp;
      reduce_total_local_addrs_temp = next; next += size_temp;
    }

    // if 'mean' and 'sum', create new space to store reduce answer.
    if ( reduction != 0 )
    {
      reduce_split_local_addrs = next; next += size;
      reduce_total_local_addrs = next; next += size;
      tpu_bdc_set_C( reduce_total_local_addrs, zero, &reduce_shape, NULL, DT_FP32 );
    }

    if ( ( int ) next <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( wmax > 1 )
      {
        wmax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT( false );
      }
    }
  }
  int todo = length;
  int done = 0;
  dim4 shape = { .n = 1, .h = 1 };
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
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
    if ( dtype != DT_FP32 )
    {
      // load systerm memory
      tpu_gdma_cpy_S2L ( input_local_addrs_temp, input_global_addr + done * dsize_global, &shape, NULL, NULL, dtype );
      // cast dtype to FP32
      tpu_bdc_cast ( input_local_addrs[index], input_local_addrs_temp, &shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
      // load systerm memory
      tpu_gdma_cpy_S2L ( input_local_addrs_temp, target_global_addr + done * dsize_global, &shape, NULL, NULL, dtype );
      // cast dtype to FP32
      tpu_bdc_cast ( target_local_addrs[index], input_local_addrs_temp, &shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
    }
    else
    {
      tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dsize_global, &shape, NULL, NULL, dtype );
      tpu_gdma_cpy_S2L ( target_local_addrs[index], target_global_addr + done * dsize_global, &shape, NULL, NULL, dtype );
    }

    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    // reduction == 'none' need to constantly change the address to output a complete Tensor, 'mean' and 'sum' just need output answer once in the end.
    if ( l2s && reduction == 0 )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    // ( out - target )^2
    tpu_bdc_fp_sub ( output_local_addrs[index], input_local_addrs[index], target_local_addrs[index], &shape, NULL, NULL, NULL, DT_FP32 );
    tpu_bdc_fp_mul ( output_local_addrs[index], output_local_addrs[index], output_local_addrs[index], &shape, NULL, NULL, NULL, DT_FP32 );
    // reduction == 'none'
    if ( reduction == 0 )
    {
      l2s_global_addr = output_global_addr + done * dsize_global;
      if ( dtype != DT_FP32 )
      {
        // cast FP32 to dtype
        tpu_bdc_cast ( output_local_addrs_temp, output_local_addrs[index], &shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
        l2s_local_addr = output_local_addrs_temp;
      }
      else
      {
        l2s_local_addr = output_local_addrs[index];
      }
      l2s_shape = shape;
    }
    // reduction == 'mean' or 'sum'
    else
    {
      // output_local_addrs[ 1, shape.c, 1, shape.w ]
      // out = REDUCE_SUM ( ( out - target )^2, [ 3 ] )
      dim2 kernel = { .h = 1, .w = shape.w };
      tpu_bdc_fp_avg_pool2d ( output_local_addrs[1 - index], output_local_addrs[index], &shape, &kernel, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, ScaleFP );

      // CW-Transpose: out[ 1, c, 1, w=1 ] = out [ 1, w=1, 1, c ]
      dim4 transpose_shape = { .n = 1, .c = 1, .h = 1, .w = shape.c };
      tpu_bdc_cw_trans( transpose_reduce_split_local_addrs, output_local_addrs[1 - index], &transpose_shape, DT_FP32);
      // out = REDUCE_SUM ( ( out - target )^2, [ 0, 1, 2, 3 ] )
      kernel.h = transpose_shape.h;
      kernel.w = transpose_shape.w;
      // reduction == 'mean'
      if ( reduction == 1)
      {
        ScaleFP.f32 = 1.f / ( shape.c * shape.w );
      }
      tpu_bdc_fp_avg_pool2d ( reduce_split_local_addrs, transpose_reduce_split_local_addrs, &transpose_shape, &kernel, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, ScaleFP );
      // total += split
      tpu_bdc_fp_add ( reduce_total_local_addrs, reduce_total_local_addrs, reduce_split_local_addrs, &reduce_shape, NULL, NULL, NULL, DT_FP32 );
    }
    l2s = true;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if ( tpu_is_parallel_state() )
  {
    tpu_parallel_end();
  }
  if ( l2s )
  {
    // reduction == 'none'
    if ( reduction == 0 ){ tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype ); }
    // reduction == 'mean' or 'sum'
    else
    {
      l2s_global_addr = output_global_addr;
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_cast ( reduce_total_local_addrs_temp, reduce_total_local_addrs, &reduce_shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
        l2s_local_addr = reduce_total_local_addrs_temp;
      }
      else
      {
       l2s_local_addr = reduce_total_local_addrs;
      }
      l2s_shape = reduce_shape;
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }

  }
}

void tpu_kernel_api_mse_loss ( const void * args )
{
  sg_api_mse_loss_t * api = ( sg_api_mse_loss_t * ) args;
  data_type_t dtype = ( data_type_t ) api->dtype;
  TPUKERNEL_ASSERT ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 );
  TPUKERNEL_ASSERT ( api->reduction == 0 || api->reduction == 1 || api->reduction == 2 );
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_mse_loss_forward (
  api->input1_global_addr,
  api->input2_global_addr,
  api->output_global_addr,
  length,
  api->reduction,
  ( data_type_t ) api->dtype);
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_mse_loss );
