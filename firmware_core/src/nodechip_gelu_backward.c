#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include <math.h>


static inline void nodechip_gelu_backward_parallel_fp16 ( global_addr_t DXGlobalAddr, global_addr_t XGlobalAddr, global_addr_t DYGlobalAddr, int Len, data_type_t dtype )
{
  local_addr_t EXPCoeffAddr = 0;
  int EXPCoeffSize = tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t EXPTableAddr = EXPCoeffAddr + EXPCoeffSize;
  int EXPTableSize = tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t ERFCoeffAddr = EXPTableAddr + EXPTableSize;
  int ERFCoeffSize = tpu_aligned_feature_size ( 1, 10, DT_FP32 );
  local_addr_t X0Addr = ERFCoeffAddr + ERFCoeffSize;
  local_addr_t X1Addr, DX0Addr, DX1Addr, DY0Addr, DY1Addr, W0Addr, W1Addr, W2Addr, W3Addr;
  local_addr_t XFP32Addr, DXFP32Addr, DYFP32Addr;
  int WMax = DIV_UP ( Len, NPU_NUM );
  while ( true )
  {
    int SizeFP32 = tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
    int SizeFP16 = tpu_aligned_feature_size ( 1, WMax, dtype );
    X1Addr = X0Addr + SizeFP16;
    DX0Addr = X1Addr + SizeFP16;
    DX1Addr = DX0Addr + SizeFP16;
    DY0Addr = DX1Addr + SizeFP16;
    DY1Addr = DY0Addr + SizeFP16;
    XFP32Addr = DY1Addr + SizeFP16;
    DYFP32Addr = XFP32Addr + SizeFP32;
    DXFP32Addr = DYFP32Addr + SizeFP32;
    W0Addr = DXFP32Addr + SizeFP32;
    W1Addr = W0Addr + SizeFP32;
    W2Addr = W1Addr + SizeFP32;
    W3Addr = W2Addr + SizeFP32;
    if ( ( int ) W3Addr + SizeFP32 <= LOCAL_MEM_SIZE )
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
  tpu_bdc_load_fp32_exp_coeff ( EXPCoeffAddr );
  tpu_bdc_load_fp32_exp_table ( EXPTableAddr );
  tpu_bdc_load_fp32_erf_coeff ( ERFCoeffAddr );
  local_addr_t XAddrs[2] = { X0Addr, X1Addr };
  local_addr_t DXAddrs[2] = { DX0Addr, DX1Addr };
  local_addr_t DYAddrs[2] = { DY0Addr, DY1Addr };
  dim4 Shape = { .n = 1, .h = 1 };
  dim4 LastShape;
  int Todo = Len, Done = 0;
  scalar_t C;
  int Index = 0;
  int LastDone = 0;
  int Count = 0;
  while ( Todo > 0 )
  {
    if ( Todo > NPU_NUM )
    {
      Shape.c = NPU_NUM;
      Shape.w = MIN ( Todo / NPU_NUM, WMax );
    }
    else
    {
      Shape.c = Todo;
      Shape.w = 1;
    }
    tpu_gdma_cpy_S2L ( XAddrs[Index], XGlobalAddr + Done * 2, &Shape, NULL, NULL, dtype );
    tpu_gdma_cpy_S2L ( DYAddrs[Index], DYGlobalAddr + Done * 2, &Shape, NULL, NULL, dtype );
    if ( Count > 0 )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( Count > 0 )
    {
      tpu_gdma_cpy_L2S ( DXGlobalAddr + LastDone * 2, DXAddrs[1 - Index], &LastShape, NULL, NULL, dtype );
    }
    // FP16 -> FP32
    tpu_bdc_cast ( XFP32Addr, XAddrs[Index], &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
    tpu_bdc_cast ( DYFP32Addr, DYAddrs[Index], &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
    // W3 = X / SQRT(2)
    C.f32 = 1 / sqrt ( 2. );
    tpu_bdc_fp_mul_C ( W3Addr, XFP32Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // DX = ERF(W3)
    tpu_bdc_fp32_erf ( DXFP32Addr, W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, ERFCoeffAddr, EXPTableAddr, &Shape );
    // DX = DX + 1
    C.f32 = 1.f;
    tpu_bdc_fp_add_C ( DXFP32Addr, DXFP32Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // DX = DX * 0.5
    C.f32 = 0.5f;
    tpu_bdc_fp_mul_C ( DXFP32Addr, DXFP32Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W0 = X * X
    tpu_bdc_fp_mul ( W0Addr, XFP32Addr, XFP32Addr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // W0 = W0 * (-0.5)
    C.f32 = -0.5f;
    tpu_bdc_fp_mul_C ( W0Addr, W0Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W3 = EXP(W0)
    tpu_bdc_fp32_exp ( W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, EXPTableAddr, &Shape );
    // W3 = W3 / SQRT(2PI)
    C.f32 = 1 / sqrt ( 2 * M_PI );
    tpu_bdc_fp_mul_C ( W3Addr, W3Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W3 = W3 * X
    tpu_bdc_fp_mul ( W3Addr, W3Addr, XFP32Addr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // DX = DX + W3
    tpu_bdc_fp_add ( DXFP32Addr, DXFP32Addr, W3Addr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // DX = DX * DY
    tpu_bdc_fp_mul ( DXFP32Addr, DXFP32Addr, DYFP32Addr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // FP32 -> FP16
    tpu_bdc_cast ( DXAddrs[Index], DXFP32Addr, &Shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
    LastDone = Done;
    LastShape = Shape;
    Todo -= Shape.c * Shape.w;
    Done += Shape.c * Shape.w;
    Index = 1 - Index;
    ++Count;
  }
  if ( Count > 0 )
  {
    tpu_parallel_end();
  }
  tpu_gdma_cpy_L2S ( DXGlobalAddr + LastDone * 2, DXAddrs[1 - Index], &LastShape, NULL, NULL, dtype );
}

static inline void nodechip_gelu_backward (
  global_addr_t DXGlobalAddr,
  global_addr_t DYGlobalAddr,
  global_addr_t XGlobalAddr,
  int           Len,
  data_type_t   dtype)
{
  const int DSize = tpu_data_type_size( dtype );
  local_addr_t EXPCoeffAddr = 0;
  int EXPCoeffSize = tpu_aligned_feature_size ( 1, dtype == DT_FP32 ? 10 : 7, dtype );
  local_addr_t ERFCoeffAddr = EXPCoeffAddr + EXPCoeffSize;
  int ERFCoeffSize = tpu_aligned_feature_size ( 1, 10, dtype );
  local_addr_t X0Addr = ERFCoeffAddr + ERFCoeffSize;
  local_addr_t X1Addr, DX0Addr, DX1Addr, DY0Addr, DY1Addr, W0Addr, W1Addr, W2Addr, W3Addr;
  int WMax = DIV_UP ( Len, NPU_NUM );
  while ( true )
  {
    int Size = tpu_aligned_feature_size ( 1, WMax, dtype );
    X1Addr = X0Addr + Size;
    DX0Addr = X1Addr  + Size;
    DX1Addr = DX0Addr  + Size;
    DY0Addr = DX1Addr + Size;
    DY1Addr = DY0Addr + Size;
    W0Addr = DY1Addr + Size;
    W1Addr = W0Addr + Size;
    W2Addr = W1Addr + Size;
    W3Addr = W2Addr + Size;
    if ( ( int ) W3Addr + Size <= LOCAL_MEM_SIZE )
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
  tpu_bdc_load_fp_exp_coeff ( EXPCoeffAddr, dtype );
  tpu_bdc_load_fp_erf_coeff ( ERFCoeffAddr, dtype );
  local_addr_t XAddrs[2] = { X0Addr, X1Addr };
  local_addr_t DXAddrs[2] = { DX0Addr, DX1Addr };
  local_addr_t DYAddrs[2] = { DY0Addr, DY1Addr };
  dim4 Shape = { .n = 1, .h = 1 };
  dim4 LastShape;
  int Todo = Len, Done = 0;
  scalar_t C;
  int Index = 0;
  int LastDone = 0;
  int Count = 0;
  while ( Todo > 0 )
  {
    if ( Todo > NPU_NUM )
    {
      Shape.c = NPU_NUM;
      Shape.w = MIN ( Todo / NPU_NUM, WMax );
    }
    else
    {
      Shape.c = Todo;
      Shape.w = 1;
    }
    tpu_gdma_cpy_S2L ( XAddrs[Index], XGlobalAddr + Done * DSize, &Shape, NULL, NULL, dtype );
    tpu_gdma_cpy_S2L ( DYAddrs[Index], DYGlobalAddr + Done * DSize, &Shape, NULL, NULL, dtype );
    if ( Count > 0 )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( Count > 0 )
    {
      tpu_gdma_cpy_L2S ( DXGlobalAddr + LastDone * DSize, DXAddrs[1 - Index], &LastShape, NULL, NULL, dtype );
    }
    // W3 = X / SQRT(2)
    C.f32 = 1 / sqrt ( 2. );
    if ( dtype != DT_FP32 ) C = tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C ( W3Addr, XAddrs[Index], C, &Shape, NULL, NULL, dtype );
    // DX = ERF(W3)
    tpu_bdc_fp_erf ( DXAddrs[Index], W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, ERFCoeffAddr, &Shape, dtype );
    // DX = DX + 1
    C.f32 = 1.f;
    if ( dtype != DT_FP32 ) C = tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_add_C ( DXAddrs[Index], DXAddrs[Index], C, &Shape, NULL, NULL, dtype );
    // DX = DX * 0.5
    C.f32 = 0.5f;
    if ( dtype != DT_FP32 ) C = tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C ( DXAddrs[Index], DXAddrs[Index], C, &Shape, NULL, NULL, dtype );
    // W0 = X * X
    tpu_bdc_fp_mul ( W0Addr, XAddrs[Index], XAddrs[Index], &Shape, NULL, NULL, NULL, dtype );
    // W0 = W0 * (-0.5)
    C.f32 = -0.5f;
    if ( dtype != DT_FP32 ) C = tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C ( W0Addr, W0Addr, C, &Shape, NULL, NULL, dtype );
    // W3 = EXP(W0)
    tpu_bdc_fp_exp ( W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, &Shape, dtype );
    // W3 = W3 / SQRT(2PI)
    C.f32 = 1 / sqrt ( 2 * M_PI );
    if ( dtype != DT_FP32 ) C = tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C ( W3Addr, W3Addr, C, &Shape, NULL, NULL, dtype );
    // W3 = W3 * X
    tpu_bdc_fp_mul ( W3Addr, W3Addr, XAddrs[Index], &Shape, NULL, NULL, NULL, dtype );
    // DX = DX + W3
    tpu_bdc_fp_add ( DXAddrs[Index], DXAddrs[Index], W3Addr, &Shape, NULL, NULL, NULL, dtype );
    // DX = DX * DY
    tpu_bdc_fp_mul ( DXAddrs[Index], DXAddrs[Index], DYAddrs[Index], &Shape, NULL, NULL, NULL, dtype );
    LastDone = Done;
    LastShape = Shape;
    Todo -= Shape.c * Shape.w;
    Done += Shape.c * Shape.w;
    Index = 1 - Index;
    ++Count;
  }
  if ( Count > 0 )
  {
    tpu_parallel_end();
  }
  tpu_gdma_cpy_L2S ( DXGlobalAddr + LastDone * DSize, DXAddrs[1 - Index], &LastShape, NULL, NULL, dtype );
}

int tpu_kernel_api_gelu_backward ( const void * args )
{
  sg_api_gelu_backward_t * api = ( sg_api_gelu_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  int Len = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    Len *= api->shape[i];
  }
  if ( api->dtype == DT_FP32 )
  {
    nodechip_gelu_backward (
      api->grad_input_global_addr,
      api->grad_output_global_addr,
      api->input_global_addr,
      Len,
      ( data_type_t ) api->dtype );
  }
  else
  {
    nodechip_gelu_backward_parallel_fp16 (
      api->grad_input_global_addr,
      api->input_global_addr,
      api->grad_output_global_addr,
      Len,
      ( data_type_t ) api->dtype );
  }

  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gelu_backward );


extern void nodechip_gelu_backward_multi_core (
  global_addr_t grad_input_global_addr,
  global_addr_t grad_output_global_addr,
  global_addr_t input_global_addr,
  int*          shape,
  int           dims,
  data_type_t   dtype,
  int           enable_8ch,
  int           input_slice_dim,
  int           output_slice_dim,
  global_addr_t* grad_input_8ch_global_addr,
  global_addr_t* grad_output_8ch_global_addr,
  global_addr_t* input_8ch_global_addr);

int tpu_kernel_api_gelu_backward_multi_core ( const void * args )
{
  sg_api_gelu_backward_t * api = ( sg_api_gelu_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
#ifdef BACKEND_SG2260
  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  nodechip_gelu_backward_multi_core (
    api->grad_input_global_addr,
    api->grad_output_global_addr,
    api->input_global_addr,
    api->shape,
    api->dim,
    ( data_type_t ) api->dtype,
    0, 0, 0, 0, 0, 0);
  tpu_poll();
  return 0;
#else
  tpu_initialize();
  int Len = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    Len *= api->shape[i];
  }
  if ( api->dtype == DT_FP32 )
  {
    nodechip_gelu_backward (
      api->grad_input_global_addr,
      api->grad_output_global_addr,
      api->input_global_addr,
      Len,
      ( data_type_t ) api->dtype );
  }
  else
  {
    nodechip_gelu_backward_parallel_fp16 (
      api->grad_input_global_addr,
      api->input_global_addr,
      api->grad_output_global_addr,
      Len,
      ( data_type_t ) api->dtype );
  }
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gelu_backward_multi_core );