#include "sg_api_struct.h"
#include "tpu_kernel.h"


static inline void nodechip_silu_forward_parallel ( global_addr_t XGlobalAddr, global_addr_t YGlobalAddr, int Len, data_type_t dtype)
{
  const int DSize = tpu_data_type_size ( dtype );
  local_addr_t EXPCoeffAddr = 0;
  int EXPCoeffSize = tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t EXPTableAddr = EXPCoeffAddr + EXPCoeffSize;
  int EXPTableSize = tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t X0Addr = EXPTableAddr + EXPTableSize;
  local_addr_t X1Addr, Y0Addr, Y1Addr, W0Addr, W1Addr;
  local_addr_t X0CastAddr, X1CastAddr, Y0CastAddr, Y1CastAddr;
  int WMax = DIV_UP ( Len, NPU_NUM );
  while ( true )
  {
    int InSize = tpu_aligned_feature_size ( 1, WMax, dtype );
    int Size = tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
    X1Addr = X0Addr + InSize;
    Y0Addr = X1Addr + InSize;
    Y1Addr = Y0Addr + InSize;
    X0CastAddr = Y1Addr + InSize;
    X1CastAddr = X0CastAddr + Size;
    Y0CastAddr = X1CastAddr + Size;
    Y1CastAddr = Y0CastAddr + Size;
    W0Addr = Y1CastAddr + Size;
    W1Addr = W0Addr + Size;
    if ( ( int ) W1Addr + Size <= LOCAL_MEM_SIZE )
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
  local_addr_t XAddrs[2] = { X0Addr, X1Addr };
  local_addr_t YAddrs[2] = { Y0Addr, Y1Addr };
  local_addr_t XCastAddrs[2] = { X0CastAddr, X1CastAddr };
  local_addr_t YCastAddrs[2] = { Y0CastAddr, Y1CastAddr };
  dim4 Shape = { .n = 1, .h = 1 };
  dim4 LastShape;
  int Todo = Len, Done = 0;
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
    if ( Count > 0 && tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( Count > 0 )
    {
      tpu_gdma_cpy_L2S ( YGlobalAddr + LastDone * DSize, YAddrs[1 - Index], &LastShape, NULL, NULL, dtype );
    }
    tpu_bdc_cast ( XCastAddrs[Index], XAddrs[Index], &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO );
    tpu_bdc_fp32_silu ( YCastAddrs[Index], XCastAddrs[Index], W0Addr, W1Addr, EXPCoeffAddr, EXPTableAddr, &Shape );
    tpu_bdc_cast ( YAddrs[Index], YCastAddrs[Index], &Shape, NULL, NULL, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO );
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
  tpu_gdma_cpy_L2S ( YGlobalAddr + LastDone * DSize, YAddrs[1 - Index], &LastShape, NULL, NULL, dtype );
}

extern void nodechip_active(
global_addr_t in_global_addr,
global_addr_t out_global_addr,
const int*    shape,
int           shape_dim,
data_type_t   dtype,
int           active_type,
float*        coef);

int tpu_kernel_api_silu(const void *args) {
  sg_api_silu_t *api = (sg_api_silu_t *) args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

if ( api->dtype != DT_FP32 )
{
    int Len = 1;
    for ( int i = 0; i < api->dim; ++i )
    {
        Len *= api->shape[i];
    }
    nodechip_silu_forward_parallel ( api->input_global_addr, api->output_global_addr, Len, api->dtype );
}
else
{
    tpu_initialize();
    nodechip_active(
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    ( data_type_t ) api->dtype,
    19,//ACTIVE_SILU
    NULL);
}
tpu_poll();
return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_silu);

#ifdef ENABLE_MULTI_CORE
int tpu_kernel_api_silu_multi_core(const void *args) {
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_silu_multi_core);
#endif