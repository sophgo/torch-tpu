#include "sg_api_struct.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

// only support (1,c,1,w) reduce c dim
static inline void nodechip_reduce_c(
  global_addr_t input_global_addr,
  global_addr_t output_global_addr,
  dim4          shape,
  int           reduce_dim,
  data_type_t   dtype )
{
  const int N = shape.n;
  const int C = shape.c;
  const int H = shape.h;
  const int W = shape.w;
  TPUKERNEL_ASSERT( dtype == DT_FP32 );
  TPUKERNEL_ASSERT( reduce_dim == 1 );
  TPUKERNEL_ASSERT( N == 1 );
  TPUKERNEL_ASSERT( H == 1 );
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  const dim4 OutputShape = { .n = N, .c = 1, .h = H, .w = W };    

  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.f };
  dim4 GlobalStride, GlobalOutStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  tpu_continuous_stride ( &GlobalOutStride, &OutputShape );

  const int DSize = tpu_data_type_size( dtype );
  /*
  *  input  : [ 1,    C,  1, WMax ]
  *  output : [ 1,    1,  1, WMax ]
  *  tmp1   : [ 1, WMax,  1,    C ]
  *  tmp2   : [ 1, WMax,  1,    1 ]
  */
  local_addr_t X0Addr = 0;
  local_addr_t X1Addr, Y0Addr, Y1Addr;
  local_addr_t Tmp1Addr, Tmp2Addr;
  int WMax = W;
  while ( true )
  {
    int XSize = DIV_UP ( C, NPU_NUM ) * tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
    X1Addr = X0Addr + XSize;
    Y0Addr = X1Addr + XSize;
    int YSize = tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
    Y1Addr = Y0Addr + YSize;
    Tmp1Addr = Y1Addr + YSize;
    int Tmp1Size = DIV_UP ( WMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, C, DT_FP32 );
    Tmp2Addr = Tmp1Addr + Tmp1Size;
    int Tmp2Size = DIV_UP ( WMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    if ( ( int ) Tmp2Addr + Tmp2Size <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( WMax > 1 )
      {
        WMax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  local_addr_t XAddrs[2] = { X0Addr, X1Addr };
  local_addr_t YAddrs[2] = { Y0Addr, Y1Addr };
  dim4 Shape = { .n = 1, .c = C, .h = 1 };
  dim4 LastShape;
  int Todo = W, Done = 0;
  int Index = 0;
  int LastDone = 0;
  int Count = 0;
  while ( Todo > 0 )
  {
    Shape.w = MIN ( Todo, WMax );
    dim4 Output_Shape = { .n = 1, .c = 1, .h = 1, .w = Shape.w };
    dim4 Trans_Shape = { .n = 1, .c = Shape.w, .h = 1, .w = C };
    tpu_gdma_cpy_S2L ( XAddrs[Index], input_global_addr + Done * DSize, &Shape, NULL, &GlobalStride, dtype );
    if ( Count > 0 && tpu_is_parallel_state())
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( Count > 0 )
    {
      tpu_gdma_cpy_L2S ( output_global_addr + LastDone * DSize, YAddrs[1 - Index], &LastShape, &GlobalOutStride, NULL, dtype );
    }
    tpu_bdc_cw_trans ( Tmp1Addr, XAddrs[Index], &Trans_Shape, dtype);
    dim2 KernelSize = { .h = 1, .w = C };
    tpu_bdc_fp_avg_pool2d ( Tmp2Addr, Tmp1Addr, &Trans_Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
    tpu_bdc_cw_trans ( YAddrs[Index], Tmp2Addr, &Output_Shape, dtype);
    LastDone = Done;
    LastShape = Output_Shape;
    Todo -= Shape.w;
    Done += Shape.w;
    Index = 1 - Index;
    ++Count;
  }
  if ( Count > 0 )
  {
    tpu_parallel_end();
  }
  tpu_gdma_cpy_L2S ( output_global_addr + LastDone * DSize, YAddrs[1 - Index], &LastShape, &GlobalOutStride, NULL, dtype );
}

void tpu_kernel_api_reduce_sum(const void *args)
{
    sg_api_reduce_sum_t *api = (sg_api_reduce_sum_t *)args;
    dim4 input_shape = { api->shape[0], api->shape[1], api->shape[2], api->shape[3]};
    tpu_initialize();
    nodechip_reduce_c(
      api->input_global_addr,
      api->output_global_addr,
      input_shape,
      api->reduce_dim,
      tpu_type_convert(api->dtype));
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_reduce_sum);