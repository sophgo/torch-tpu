#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

static inline void nodechip_gelu_forward_parallel ( global_addr_t XGlobalAddr, global_addr_t YGlobalAddr, int Len )
{
  const int DSize = tpu_data_type_size ( DT_FP16 );
  local_addr_t EXPCoeffAddr = 0;
  int EXPCoeffSize = tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  local_addr_t EXPTableAddr = EXPCoeffAddr + EXPCoeffSize;
  int EXPTableSize = tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  local_addr_t ERFCoeffAddr = EXPTableAddr + EXPTableSize;
  int ERFCoeffSize = tpu_aligned_feature_size ( 1, 10, DT_FP32 );
  local_addr_t X0Addr = ERFCoeffAddr + ERFCoeffSize;
  local_addr_t X1Addr, Y0Addr, Y1Addr, W0Addr, W1Addr, W2Addr, W3Addr;
  local_addr_t X0CastAddr, X1CastAddr, Y0CastAddr, Y1CastAddr;
  int WMax = DIV_UP ( Len, NPU_NUM );
  while ( true )
  {
    int InSize = tpu_aligned_feature_size ( 1, WMax, DT_FP16 );
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
    tpu_gdma_cpy_S2L ( XAddrs[Index], XGlobalAddr + Done * DSize, &Shape, NULL, NULL, DT_FP16 );
    if ( Count > 0 && tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( Count > 0 )
    {
      tpu_gdma_cpy_L2S ( YGlobalAddr + LastDone * DSize, YAddrs[1 - Index], &LastShape, NULL, NULL, DT_FP16 );
    }
    tpu_bdc_cast ( XCastAddrs[Index], XAddrs[Index], &Shape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
    tpu_bdc_fp32_gelu ( YCastAddrs[Index], XCastAddrs[Index], W0Addr, W1Addr, W2Addr, W3Addr, EXPCoeffAddr, ERFCoeffAddr, EXPTableAddr, &Shape );
    tpu_bdc_cast ( YAddrs[Index], YCastAddrs[Index], &Shape, NULL, NULL, DT_FP16, DT_FP32, RM_HALF_TO_EVEN );
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
  tpu_gdma_cpy_L2S ( YGlobalAddr + LastDone * DSize, YAddrs[1 - Index], &LastShape, NULL, NULL, DT_FP16 );
}

extern void nodechip_active(
global_addr_t in_global_addr,
global_addr_t out_global_addr,
const int*    shape,
int           shape_dim,
data_type_t   dtype,
int           active_type,
float*        coef);

void tpu_kernel_api_gelu ( const void * args )
{
  sg_api_gelu_t * api = ( sg_api_gelu_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  if ( api->dtype != DT_FP32 )
  {
    int Len = 1;
    for ( int i = 0; i < api->dim; ++i )
    {
      Len *= api->shape[i];
    }
    nodechip_gelu_forward_parallel ( api->input_global_addr, api->output_global_addr, Len );
  }
  else
  {
    nodechip_active(
      api->input_global_addr,
      api->output_global_addr,
      api->shape,
      api->dim,
      ( data_type_t ) api->dtype,
      29,//ACTIVE_GELU
      NULL);
  }
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gelu );

#ifdef FIRMWARE_BACKEND_2260
extern void nodechip_gelu_forward_multi_core (
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int*          shape,
int           dims,
data_type_t   dtype );

void tpu_kernel_api_gelu_multi_core ( const void * args )
{
  sg_api_gelu_t * api = ( sg_api_gelu_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
#ifdef USING_PERF_MODE
    tpu_sync_all();
#endif
  nodechip_gelu_forward_multi_core (
    api->input_global_addr,
    api->output_global_addr,
    api->shape,
    api->dim,
    ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gelu_multi_core );
#endif