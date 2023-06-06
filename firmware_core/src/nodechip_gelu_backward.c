#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#include <math.h>

static inline void nodechip_gelu_backward ( global_addr_t DXGlobalAddr, global_addr_t XGlobalAddr, global_addr_t DYGlobalAddr, int Len )
{
  local_addr_t EXPCoeffAddr = 0;
  int EXPCoeffSize = tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t EXPTableAddr = EXPCoeffAddr + EXPCoeffSize;
  int EXPTableSize = tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t ERFCoeffAddr = EXPTableAddr + EXPTableSize;
  int ERFCoeffSize = tpu_aligned_feature_size ( 1, 10, DT_FP32 );
  local_addr_t XAddr = ERFCoeffAddr + ERFCoeffSize;
  local_addr_t DXAddr, DYAddr, W0Addr, W1Addr, W2Addr, W3Addr;
  int WMax = DIV_UP ( Len, NPU_NUM );
  while ( true )
  {
    int Size = tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
    DXAddr = XAddr  + Size;
    DYAddr = DXAddr + Size;
    W0Addr = DYAddr + Size;
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
  tpu_bdc_load_fp32_exp_coeff ( EXPCoeffAddr );
  tpu_bdc_load_fp32_exp_table ( EXPTableAddr );
  tpu_bdc_load_fp32_erf_coeff ( ERFCoeffAddr );
  dim4 Shape = { .n = 1, .h = 1 };
  int Todo = Len, Done = 0;
  scalar_t C;
  while ( Todo > 0 )
  {
    if ( Todo > NPU_NUM )
    {
      Shape.c = NPU_NUM;
      Shape.w = MIN ( Todo / NPU_NUM, WMax );
    }
    else
    {
      Shape.c = 1;
      Shape.w = Todo;
    }
    tpu_gdma_cpy_S2L ( XAddr, XGlobalAddr + Done * sizeof ( float ), &Shape, NULL, NULL, DT_FP32 );
    tpu_gdma_cpy_S2L ( DYAddr, DYGlobalAddr + Done * sizeof ( float ), &Shape, NULL, NULL, DT_FP32 );
    // W3 = X / SQRT(2)
    C.f32 = 1 / sqrt ( 2. );
    tpu_bdc_fp_mul_C ( W3Addr, XAddr, C, &Shape, NULL, NULL, DT_FP32 );
    // DX = ERF(W3)
    tpu_bdc_fp32_erf ( DXAddr, W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, ERFCoeffAddr, EXPTableAddr, &Shape );
    // DX = DX + 1
    C.f32 = 1.f;
    tpu_bdc_fp_add_C ( DXAddr, DXAddr, C, &Shape, NULL, NULL, DT_FP32 );
    // DX = DX * 0.5
    C.f32 = 0.5f;
    tpu_bdc_fp_mul_C ( DXAddr, DXAddr, C, &Shape, NULL, NULL, DT_FP32 );
    // W0 = X * X
    tpu_bdc_fp_mul ( W0Addr, XAddr, XAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // W0 = W0 * (-0.5)
    C.f32 = -0.5f;
    tpu_bdc_fp_mul_C ( W0Addr, W0Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W3 = EXP(W0)
    tpu_bdc_fp32_exp ( W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, EXPTableAddr, &Shape );
    // W3 = W3 / SQRT(2PI)
    C.f32 = 1 / sqrt ( 2 * M_PI );
    tpu_bdc_fp_mul_C ( W3Addr, W3Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W3 = W3 * X
    tpu_bdc_fp_mul ( W3Addr, W3Addr, XAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // DX = DX + W3
    tpu_bdc_fp_add ( DXAddr, DXAddr, W3Addr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // DX = DX * DY
    tpu_bdc_fp_mul ( DXAddr, DXAddr, DYAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
    tpu_gdma_cpy_L2S ( DXGlobalAddr + Done * sizeof ( float ), DXAddr, &Shape, NULL, NULL, DT_FP32 );
    Todo -= Shape.c * Shape.w;
    Done += Shape.c * Shape.w;
  }
}

static inline void nodechip_gelu_backward_parallel ( global_addr_t DXGlobalAddr, global_addr_t XGlobalAddr, global_addr_t DYGlobalAddr, int Len )
{
  local_addr_t EXPCoeffAddr = 0;
  int EXPCoeffSize = tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t EXPTableAddr = EXPCoeffAddr + EXPCoeffSize;
  int EXPTableSize = tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t ERFCoeffAddr = EXPTableAddr + EXPTableSize;
  int ERFCoeffSize = tpu_aligned_feature_size ( 1, 10, DT_FP32 );
  local_addr_t X0Addr = ERFCoeffAddr + ERFCoeffSize;
  local_addr_t X1Addr, DX0Addr, DX1Addr, DY0Addr, DY1Addr, W0Addr, W1Addr, W2Addr, W3Addr;
  int WMax = DIV_UP ( Len, NPU_NUM );
  while ( true )
  {
    int Size = tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
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
    tpu_gdma_cpy_S2L ( XAddrs[Index], XGlobalAddr + Done * sizeof ( float ), &Shape, NULL, NULL, DT_FP32 );
    tpu_gdma_cpy_S2L ( DYAddrs[Index], DYGlobalAddr + Done * sizeof ( float ), &Shape, NULL, NULL, DT_FP32 );
    if ( Count > 0 )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( Count > 0 )
    {
      tpu_gdma_cpy_L2S ( DXGlobalAddr + LastDone * sizeof ( float ), DXAddrs[1 - Index], &LastShape, NULL, NULL, DT_FP32 );
    }
    // W3 = X / SQRT(2)
    C.f32 = 1 / sqrt ( 2. );
    tpu_bdc_fp_mul_C ( W3Addr, XAddrs[Index], C, &Shape, NULL, NULL, DT_FP32 );
    // DX = ERF(W3)
    tpu_bdc_fp32_erf ( DXAddrs[Index], W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, ERFCoeffAddr, EXPTableAddr, &Shape );
    // DX = DX + 1
    C.f32 = 1.f;
    tpu_bdc_fp_add_C ( DXAddrs[Index], DXAddrs[Index], C, &Shape, NULL, NULL, DT_FP32 );
    // DX = DX * 0.5
    C.f32 = 0.5f;
    tpu_bdc_fp_mul_C ( DXAddrs[Index], DXAddrs[Index], C, &Shape, NULL, NULL, DT_FP32 );
    // W0 = X * X
    tpu_bdc_fp_mul ( W0Addr, XAddrs[Index], XAddrs[Index], &Shape, NULL, NULL, NULL, DT_FP32 );
    // W0 = W0 * (-0.5)
    C.f32 = -0.5f;
    tpu_bdc_fp_mul_C ( W0Addr, W0Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W3 = EXP(W0)
    tpu_bdc_fp32_exp ( W3Addr, W0Addr, W1Addr, W2Addr, EXPCoeffAddr, EXPTableAddr, &Shape );
    // W3 = W3 / SQRT(2PI)
    C.f32 = 1 / sqrt ( 2 * M_PI );
    tpu_bdc_fp_mul_C ( W3Addr, W3Addr, C, &Shape, NULL, NULL, DT_FP32 );
    // W3 = W3 * X
    tpu_bdc_fp_mul ( W3Addr, W3Addr, XAddrs[Index], &Shape, NULL, NULL, NULL, DT_FP32 );
    // DX = DX + W3
    tpu_bdc_fp_add ( DXAddrs[Index], DXAddrs[Index], W3Addr, &Shape, NULL, NULL, NULL, DT_FP32 );
    // DX = DX * DY
    tpu_bdc_fp_mul ( DXAddrs[Index], DXAddrs[Index], DYAddrs[Index], &Shape, NULL, NULL, NULL, DT_FP32 );
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
  tpu_gdma_cpy_L2S ( DXGlobalAddr + LastDone * sizeof ( float ), DXAddrs[1 - Index], &LastShape, NULL, NULL, DT_FP32 );
}

void tpu_kernel_api_gelu_backward ( const void * args )
{
  sg_api_gelu_backward_t * api = ( sg_api_gelu_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == SG_DTYPE_FP32 );
  int Len = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    Len *= api->shape[i];
  }
  tpu_initialize();
  nodechip_gelu_backward_parallel ( api->dx_global_addr, api->x_global_addr, api->dy_global_addr, Len );
  tpu_poll();
}
