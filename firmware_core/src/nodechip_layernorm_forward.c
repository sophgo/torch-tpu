#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#include "tpu_utils.h"

static inline void nodechip_layernorm_forward (
  global_addr_t input_global_addr,
  global_addr_t weight_global_addr,
  global_addr_t bias_global_addr,
  global_addr_t output_global_addr,
  global_addr_t mean_global_addr,
  global_addr_t rstd_global_addr,
  const int*    shape,
  int           dims,
  int           axis,
  float         eps,
  bool          affine,
  bool          save_stat,
  data_type_t   dtype )
{
  const bool weight_defined = (weight_global_addr != 0 && affine);
  const bool bias_defined = (bias_global_addr != 0 && affine);
  int inner_num = 1, outer_num = 1;
  for (int i = 0; i < axis; ++i)
  {
    outer_num *= shape[i];
  }
  for (int i = axis; i < dims; ++i)
  {
    inner_num *= shape[i];
  }
  const int N = 1;
  const int C = outer_num;
  const int H = 1;
  const int W = inner_num;
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const int CPerNPU = DIV_UP ( C, NPU_NUM );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t InvInner = { .f32 = 1.0 / ( ( double ) inner_num ) };
  const scalar_t EPS = { .f32 = eps };
  const int DataSize = tpu_data_type_size ( DT_FP32 );
  const int HalfSize = tpu_data_type_size ( DT_FP16 );
  /*
   *  In    : [ 1,  CMax,   1,  WMax ]
   *  Out   : [ 1,  CMax,   1,  WMax ]
   *  Tmp   : [ 1,  CMax,   1,     1 ]
   *  Mean  : [ 1,  CMax,   1,     1 ]
   *  Rstd  : [ 1,  CMax,   1,     1 ]
   *  Weight: [ 1,   NPU,   1,  WMax ]
   *  Bias  : [ 1,   NPU,   1,  WMax ]
   */
  int CMax = CPerNPU * NPU_NUM, WMax = W;
  local_addr_t ICast, OCast, MVCast, WBCast;
  local_addr_t IAddr, OAddr, TAddr, MAddr, VAddr, WAddr, BAddr;
  int CastSize, MVCastSize, WBCastSize;
  bool Split = false;
  while ( true )
  {
    Split = CMax != C || WMax != W ;
    IAddr = 0;
    if( dtype == DT_FP16 )
    {
      CastSize = tpu_aligned_feature_size ( 1, WMax, DT_FP16 ) * DIV_UP ( CMax, NPU_NUM );
      MVCastSize = tpu_aligned_feature_size ( 1, 1, DT_FP16 ) * DIV_UP ( CMax, NPU_NUM );
      WBCastSize = tpu_aligned_feature_size ( 1, WMax, DT_FP16 );
      ICast = 0;
      OCast = ICast + CastSize;
      MVCast = OCast + CastSize;
      WBCast = MVCast + MVCastSize;
      IAddr = WBCast + WBCastSize;
    }
    int ISize = tpu_aligned_feature_size ( 1, WMax, DT_FP32 ) * DIV_UP ( CMax, NPU_NUM );
    OAddr = IAddr + ISize;
    int OSize = tpu_aligned_feature_size ( 1, WMax, DT_FP32 ) * DIV_UP ( CMax, NPU_NUM );
    TAddr = OAddr + OSize;
    int TSize = tpu_aligned_feature_size ( 1, 1, DT_FP32 ) * DIV_UP ( CMax, NPU_NUM );
    MAddr = TAddr + TSize;
    int MSize = tpu_aligned_feature_size ( 1, 1, DT_FP32 ) * DIV_UP ( CMax, NPU_NUM );
    VAddr = MAddr + MSize;
    int VSize = tpu_aligned_feature_size ( 1, 1, DT_FP32 ) * DIV_UP ( CMax, NPU_NUM );
    WAddr = VAddr + VSize;
    int WSize = tpu_aligned_feature_size ( 1, WMax, DT_FP32 );
    BAddr = WAddr + WSize;
    int BSize = WSize;
    if ( ( int ) BAddr + BSize <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( CMax > NPU_NUM )
      {
        CMax -= NPU_NUM;
        continue;
      }
      else if ( WMax > 1 )
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
  TPUKERNEL_DBG("Split:%d\n", Split);
  dim4 Shape = { .n = 1, .h = 1 };
  int CTodo = C, CDone = 0;
  while ( CTodo > 0 )
  {
    Shape.c = MIN ( CTodo, CMax );
    dim4 CShape = { .n = 1, .c = Shape.c, .h = 1, .w = 1 };
    dim4 CStride;
    tpu_aligned_stride ( &CStride, 0, &CShape, DT_FP32 );
    dim4 CBcastStride = { .n = 0, .c = CStride.c, .h = 0, .w = 0 };

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Mean */
    int WTodo = W, WDone = 0;
    while ( WTodo > 0 )
    {
      Shape.w = MIN ( WTodo, WMax );
      dim2 KernelSize = { .h = 1, .w = Shape.w };
      /* Move input from global to local */
      if ( dtype == DT_FP16 )
      {
        global_addr_t IGAddr = input_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * HalfSize;
        tpu_gdma_cpy_S2L ( ICast, IGAddr, &Shape, NULL, &GlobalStride, DT_FP16 );
        tpu_bdc_cast ( IAddr, ICast, &Shape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
      }
      else
      {
        global_addr_t IGAddr = input_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * DataSize;
        tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, DT_FP32 );
      }
      if ( WDone == 0 )
      {
        tpu_bdc_fp_avg_pool2d ( MAddr, IAddr, &Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, InvInner );
      }
      else
      {
        tpu_bdc_fp_avg_pool2d ( TAddr, IAddr, &Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, InvInner );
        tpu_bdc_fp_add ( MAddr, MAddr, TAddr, &CShape, NULL, NULL, NULL, DT_FP32 );
      }
      WTodo -= Shape.w;
      WDone += Shape.w;
    }
    /* Move MEAN from local to global memory */
    if ( dtype == DT_FP16 )
    {
      tpu_bdc_cast ( MVCast, MAddr, &CShape, NULL, NULL, DT_FP16, DT_FP32, RM_HALF_TO_EVEN );
      tpu_gdma_cpy_L2S ( mean_global_addr + CDone * HalfSize, MVCast, &CShape, NULL, NULL, DT_FP16 );
    }
    else
    {
      tpu_gdma_cpy_L2S ( mean_global_addr + CDone * DataSize, MAddr, &CShape, NULL, NULL, DT_FP32 );
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Var */
    WTodo = W, WDone = 0;
    while ( WTodo > 0 )
    {
      Shape.w = MIN ( WTodo, WMax );
      dim2 KernelSize = { .h = 1, .w = Shape.w };
      if ( Split == true )
      {
        if ( dtype == DT_FP16 )
        {
          global_addr_t IGAddr = input_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * HalfSize;
          tpu_gdma_cpy_S2L ( ICast, IGAddr, &Shape, NULL, &GlobalStride, DT_FP16 );
          tpu_bdc_cast ( IAddr, ICast, &Shape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
        }
        else
        {
          global_addr_t IGAddr = input_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * DataSize;
          tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, DT_FP32 );
        }
      }
      tpu_bdc_fp_sub ( OAddr, IAddr, MAddr, &Shape, NULL, NULL, &CBcastStride, DT_FP32 );
      tpu_bdc_fp_mul ( OAddr, OAddr, OAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
      if ( WDone == 0 )
      {
        tpu_bdc_fp_avg_pool2d ( VAddr, OAddr, &Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, InvInner );
      }
      else
      {
        tpu_bdc_fp_avg_pool2d ( TAddr, OAddr, &Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, InvInner );
        tpu_bdc_fp_add ( VAddr, VAddr, TAddr, &CShape, NULL, NULL, NULL, DT_FP32 );
      }
      WTodo -= Shape.w;
      WDone += Shape.w;
    }
    tpu_bdc_fp_add_C ( TAddr, VAddr, EPS, &CShape, NULL, NULL, DT_FP32 );
    tpu_bdc_fp32_rsqrt ( VAddr, TAddr, &CShape );
    /* Move RSTD from local to global memory */
    if ( dtype == DT_FP16 )
    {
      tpu_bdc_cast ( MVCast, VAddr, &CShape, NULL, NULL, DT_FP16, DT_FP32, RM_HALF_TO_EVEN );
      tpu_gdma_cpy_L2S ( rstd_global_addr + CDone * HalfSize, MVCast, &CShape, NULL, NULL, DT_FP16 );
    }
    else
    {
      tpu_gdma_cpy_L2S ( rstd_global_addr + CDone * DataSize, VAddr, &CShape, NULL, NULL, DT_FP32 );
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Output */
    WTodo = W, WDone = 0;
    while ( WTodo > 0 )
    {
      Shape.w = MIN ( WTodo, WMax );
      dim4 WShape = { .n = 1, .c = 1, .h = 1, .w = Shape.w };
      dim4 WBCastShape = { .n = 1, .c = NPU_NUM, .h = 1, .w = Shape.w };
      dim4 WBcastStride;
      tpu_aligned_stride ( &WBcastStride, 0, &WBCastShape, DT_FP32 );
      WBcastStride.c = 0;
      if ( Split == true )
      {
        if ( dtype == DT_FP16 )
        {
          global_addr_t IGAddr = input_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * HalfSize;
          tpu_gdma_cpy_S2L ( ICast, IGAddr, &Shape, NULL, &GlobalStride, DT_FP16 );
          tpu_bdc_cast ( IAddr, ICast, &Shape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
        }
        else
        {
          global_addr_t IGAddr = input_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * DataSize;
          tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, DT_FP32 );
        }
      }
      tpu_bdc_fp_sub ( OAddr, IAddr, MAddr, &Shape, NULL, NULL, &CBcastStride, DT_FP32 );
      tpu_bdc_fp_mul ( OAddr, OAddr, VAddr, &Shape, NULL, NULL, &CBcastStride, DT_FP32 );
      if ( weight_defined )
      {
        if ( dtype == DT_FP16 )
        {
          tpu_gdma_cpy_S2L ( WBCast, weight_global_addr + WDone * HalfSize, &WShape, NULL, NULL, DT_FP16 );
          tpu_bdc_cast ( WAddr, WBCast, &WShape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
          tpu_bdc_npu_bcast ( WAddr, WAddr, &WBCastShape, DT_FP32 );
        }
        else
        {
          tpu_gdma_cpy_S2L ( WAddr, weight_global_addr + WDone * DataSize, &WShape, NULL, NULL, DT_FP32 );
          tpu_bdc_npu_bcast ( WAddr, WAddr, &WBCastShape, DT_FP32 );
        }
        tpu_bdc_fp_mul ( OAddr, OAddr, WAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      }
      if ( bias_defined )
      {
        if ( dtype == DT_FP16 )
        {
          tpu_gdma_cpy_S2L ( WBCast, bias_global_addr + WDone * HalfSize, &WShape, NULL, NULL, DT_FP16 );
          tpu_bdc_cast ( BAddr, WBCast, &WShape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
          tpu_bdc_npu_bcast ( BAddr, BAddr, &WBCastShape, DT_FP32 );
        }
        else
        {
          tpu_gdma_cpy_S2L ( BAddr, bias_global_addr + WDone * DataSize, &WShape, NULL, NULL, DT_FP32 );
          tpu_bdc_npu_bcast ( BAddr, BAddr, &WBCastShape, DT_FP32 );
        }
        tpu_bdc_fp_add ( OAddr, OAddr, BAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      }
      /* Move output from local to global */
      if ( dtype == DT_FP16 )
      {
        global_addr_t OGAddr = output_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * HalfSize;
        tpu_bdc_cast ( OCast, OAddr, &Shape, NULL, NULL, DT_FP16, DT_FP32, RM_HALF_TO_EVEN );
        tpu_gdma_cpy_L2S ( OGAddr, OCast, &Shape, &GlobalStride, NULL, DT_FP16 );
      }
      else
      {
        global_addr_t OGAddr = output_global_addr + ( WDone * GlobalStride.w + CDone * GlobalStride.c ) * DataSize;
        tpu_gdma_cpy_L2S ( OGAddr, OAddr, &Shape, &GlobalStride, NULL, DT_FP32 );
      }
      WTodo -= Shape.w;
      WDone += Shape.w;
    }
    CTodo -= Shape.c;
    CDone += Shape.c;
  }
}

void tpu_kernel_api_layernorm_forward(const void *args)
{
    sg_api_layernorm_forward_t *api = (sg_api_layernorm_forward_t *)args;

    tpu_initialize();
    nodechip_layernorm_forward(
      api->input_global_addr,
      api->weight_global_addr,
      api->bias_global_addr,
      api->output_global_addr,
      api->mean_global_addr,
      api->rstd_global_addr,
      api->shape,
      api->dims,
      api->axis,
      api->eps,
      api->affine,
      api->save_stat,
      tpu_type_convert(api->dtype));
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_layernorm_forward);