#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

static inline void nodechip_batchnorm2d_forward_training_parallel (
global_addr_t input_global_addr,
global_addr_t running_mean_global_addr,
global_addr_t running_var_global_addr,
global_addr_t weight_global_addr,
global_addr_t bias_global_addr,
global_addr_t updated_mean_global_addr,
global_addr_t updated_var_global_addr,
global_addr_t batch_mean_global_addr,
global_addr_t batch_invstd_global_addr,
global_addr_t output_global_addr,
dim4          shape,
float         momentum,
float         eps,
data_type_t   dtype )
{
  const bool weight_defined = weight_global_addr != 0;
  const bool bias_defined = bias_global_addr != 0;
  const bool running_mean_defined = running_mean_global_addr != 0;
  const bool running_var_defined = running_var_global_addr != 0;
  const int N = shape.n;
  const int C = shape.c;
  const int H = shape.h;
  const int W = shape.w;
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const int CPerNPU = DIV_UP ( C, NPU_NUM );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.f };
  const scalar_t InvNHW = { .f32 = 1.0 / ( ( double ) N * H * W ) };
  const scalar_t InvNHWMinusOne = { .f32 = 1.0 / ( ( double ) N * H * W - 1 ) };
  const scalar_t EPS = { .f32 = eps };
  const scalar_t Momentum = { .f32 = momentum };
  const scalar_t OneMinusMomentum = { .f32 = 1.f - momentum };
  const int DataSize = tpu_data_type_size ( dtype );
  /*
   *  I: [ NMax, CMax, HMax,     W ]
   *  O: [ NMax, CMax, HMax,     W ]
   *  P: [ NMax, CMax, HMax,     1 ]
   *  T: [ NMax, CMax,    1,     1 ]
   *  S: [ 1,    CMax,    1,  NMax ]
   *  M: [ 1,    CMax,    1,     1 ]
   *  V: [ 1,    CMax,    1,     1 ]
   */
  int HMax = H, NMax = N, CMax = CPerNPU * NPU_NUM;
  local_addr_t IAddr, OAddr, PAddr, TAddr, SAddr, MAddr, VAddr;
  bool Split = false;
  while ( true )
  {
    Split = NMax != N || HMax != H ;
    IAddr = 0;
    int ISize = tpu_aligned_feature_size ( HMax, W, dtype ) * DIV_UP ( CMax, NPU_NUM ) * NMax;
    OAddr = IAddr + ISize;
    int OSize = ISize;
    PAddr = OAddr + OSize;
    int PSize = tpu_aligned_feature_size ( HMax, 1, dtype ) * DIV_UP ( CMax, NPU_NUM ) * NMax;
    TAddr = PAddr + PSize;
    int TSize = tpu_aligned_feature_size ( 1, 1, dtype ) * DIV_UP ( CMax, NPU_NUM ) * NMax;
    SAddr = TAddr + TSize;
    int SSize = tpu_aligned_feature_size ( 1, NMax, dtype ) * DIV_UP ( CMax, NPU_NUM );
    MAddr = SAddr + SSize;
    int MSize = tpu_aligned_feature_size ( 1,  1, dtype ) * DIV_UP ( CMax, NPU_NUM );
    VAddr = MAddr + MSize;
    int VSize = MSize;
    if ( ( int ) VAddr + VSize <= LOCAL_MEM_SIZE )
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
      else if ( NMax > 1 )
      {
        NMax /= 2;
        continue;
      }
      else if ( HMax > 1 )
      {
        HMax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  dim4 Shape = { .w = W };
  local_addr_t IOAddr[2] = { IAddr, OAddr };
  int CTodo = C, CDone = 0;
  while ( CTodo > 0 )
  {
    Shape.c = MIN ( CTodo, CMax );
    const dim4 CShape = { .n = 1, .c = Shape.c, .h = 1, .w = 1 };
    dim4 CStride;
    tpu_aligned_stride ( &CStride, 0, &CShape, dtype );
    dim4 CBcastStride = { .n = 0, .c = CStride.c, .h = 0, .w = 0 };
    int IOIndex = 0;
    /* Compute Mean */
    int NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = 1 };
      dim4 SShape = { .n = 1, .c = Shape.c, .h = 1, .w = Shape.n };
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        /* Move input from global to local */
        global_addr_t IGAddr = input_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
        if ( Split == true )
        {
          tpu_gdma_cpy_S2L ( IOAddr[IOIndex], IGAddr, &Shape, NULL, &GlobalStride, dtype );
          if ( NDone > 0 || HDone > 0 )
          {
            tpu_parallel_end();
          }
          tpu_parallel_start();
        }
        else
        {
          tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, dtype );
        }
        /* P = REDUCE_SUM ( I, [ 3 ] ) */
        dim2 KernelSizeDim3 = { .h = 1, .w = Shape.w };
        if ( Split == true )
        {
          tpu_bdc_fp_avg_pool2d ( PAddr, IOAddr[IOIndex], &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        }
        else
        {
          tpu_bdc_fp_avg_pool2d ( PAddr, IAddr, &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        }
        /* T = REDUCE_SUM ( P, [ 2 ] ) */
        dim2 KernelSizeDim2 = { .h = Shape.h, .w = 1 };
        dim4 PShape = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
        tpu_bdc_fp_avg_pool2d ( TAddr, PAddr, &PShape, &KernelSizeDim2, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        dim4 TStride;
        tpu_aligned_stride ( &TStride, 0, &TShape, dtype );
        dim4 TTransStride = { .n = TStride.w, .c = TStride.c, .h = TStride.h, .w = TStride.n };
        if ( HDone == 0 )
        {
          /* S = NW-Transpose ( T ) */
          tpu_bdc_cpy ( SAddr, TAddr, &SShape, NULL, &TTransStride, dtype );
        }
        else
        {
          /* S = S + NW-Transpose ( T ) */
          tpu_bdc_fp_add ( SAddr, SAddr, TAddr, &SShape, NULL, NULL, &TTransStride, dtype );
        }
        HTodo -= Shape.h;
        HDone += Shape.h;
        IOIndex = 1 - IOIndex;
      }
      /* T or V = REDUCE_SUM ( S, [ 2, 3 ] ) */
      dim2 KernelSize = { .h = SShape.h, .w = SShape.w };
      tpu_bdc_fp_avg_pool2d ( NDone == 0 ? VAddr : TAddr, SAddr, &SShape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( NDone > 0 )
      {
        /* V = V + T */
        tpu_bdc_fp_add ( VAddr, VAddr, TAddr, &CShape, NULL, NULL, NULL, dtype );
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    if ( Split == true )
    {
      tpu_parallel_end();
    }
    /* M = V / ( N * H * W ) */
    tpu_bdc_fp_mul_C ( MAddr, VAddr, InvNHW, &CShape, NULL, NULL, dtype );
    /* Move M as MEAN from local to global memory */
    tpu_gdma_cpy_L2S ( batch_mean_global_addr + CDone * DataSize, MAddr, &CShape, NULL, NULL, dtype );
    if ( running_mean_defined )
    {
      /* Move running_mean from global to local memory */
      tpu_gdma_cpy_S2L ( TAddr, running_mean_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
      /* S = M * Momentum */
      tpu_bdc_fp_mul_C ( SAddr, MAddr, Momentum, &CShape, NULL, NULL, dtype );
      /* T = T * ( 1 - Momentum ) */
      tpu_bdc_fp_mul_C ( TAddr, TAddr, OneMinusMomentum, &CShape, NULL, NULL, dtype );
      /* T = T + S */
      tpu_bdc_fp_add ( TAddr, TAddr, SAddr, &CShape, NULL, NULL, NULL, dtype );
      /* Move T as UPDATED RUNNING MEAN from local to global memory */
      tpu_gdma_cpy_L2S ( updated_mean_global_addr + CDone * DataSize, TAddr, &CShape, NULL, NULL, dtype );
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Var */
    IOIndex = 0;
    NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = 1 };
      dim4 SShape = { .n = 1, .c = Shape.c, .h = 1, .w = Shape.n };
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        if ( Split == true )
        {
          global_addr_t IGAddr = input_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
          tpu_gdma_cpy_S2L ( IOAddr[IOIndex], IGAddr, &Shape, NULL, &GlobalStride, dtype );
          if ( NDone > 0 || HDone > 0 )
          {
            tpu_parallel_end();
          }
          tpu_parallel_start();
        }
        if ( Split == true )
        {
          /* O = I - M */
          tpu_bdc_fp_sub ( IOAddr[IOIndex], IOAddr[IOIndex], MAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          /* O = O * O */
          tpu_bdc_fp_mul ( IOAddr[IOIndex], IOAddr[IOIndex], IOAddr[IOIndex], &Shape, NULL, NULL, NULL, dtype );
          /* P = REDUCE_SUM ( O, [ 3 ] ) */
          dim2 KernelSizeDim3 = { .h = 1, .w = Shape.w };
          tpu_bdc_fp_avg_pool2d ( PAddr, IOAddr[IOIndex], &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        }
        else
        {
          /* O = I - M */
          tpu_bdc_fp_sub ( OAddr, IAddr, MAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          /* O = O * O */
          tpu_bdc_fp_mul ( OAddr, OAddr, OAddr, &Shape, NULL, NULL, NULL, dtype );
          /* P = REDUCE_SUM ( O, [ 3 ] ) */
          dim2 KernelSizeDim3 = { .h = 1, .w = Shape.w };
          tpu_bdc_fp_avg_pool2d ( PAddr, OAddr, &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        }
        /* T = REDUCE_SUM ( P, [ 2 ] ) */
        dim2 KernelSizeDim2 = { .h = Shape.h, .w = 1 };
        dim4 PShape = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
        tpu_bdc_fp_avg_pool2d ( TAddr, PAddr, &PShape, &KernelSizeDim2, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        dim4 TStride;
        tpu_aligned_stride ( &TStride, 0, &TShape, dtype );
        dim4 TTransStride = { .n = TStride.w, .c = TStride.c, .h = TStride.h, .w = TStride.n };
        if ( HDone == 0 )
        {
          /* S = NW-Transpose ( T ) */
          tpu_bdc_cpy ( SAddr, TAddr, &SShape, NULL, &TTransStride, dtype );
        }
        else
        {
          /* S = S + NW-Transpose ( T ) */
          tpu_bdc_fp_add ( SAddr, SAddr, TAddr, &SShape, NULL, NULL, &TTransStride, dtype );
        }
        HTodo -= Shape.h;
        HDone += Shape.h;
        IOIndex = 1 - IOIndex;
      }
      /* T or V = REDUCE_SUM ( S, [ 2, 3 ] ) */
      dim2 KernelSize = { .h = SShape.h, .w = SShape.w };
      tpu_bdc_fp_avg_pool2d ( NDone == 0 ? VAddr : TAddr, SAddr, &SShape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( NDone > 0 )
      {
        /* V = V + T */
        tpu_bdc_fp_add ( VAddr, VAddr, TAddr, &CShape, NULL, NULL, NULL, dtype );
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    if ( Split == true )
    {
      tpu_parallel_end();
    }
    if ( running_var_defined )
    {
      /* Move running_var from global to local memory */
      tpu_gdma_cpy_S2L ( PAddr, running_var_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
      /* T = V / ( N * H * W - 1 ) */
      tpu_bdc_fp_mul_C ( TAddr, VAddr, InvNHWMinusOne, &CShape, NULL, NULL, dtype );
      /* S = T * Momentum */
      tpu_bdc_fp_mul_C ( SAddr, TAddr, Momentum, &CShape, NULL, NULL, dtype );
      /* P = P * ( 1 - Momentum ) */
      tpu_bdc_fp_mul_C ( PAddr, PAddr, OneMinusMomentum, &CShape, NULL, NULL, dtype );
      /* S = S + P */
      tpu_bdc_fp_add ( SAddr, SAddr, PAddr, &CShape, NULL, NULL, NULL, dtype );
      /* Move T as UPDATED RUNNING VAR from local to global memory */
      tpu_gdma_cpy_L2S ( updated_var_global_addr + CDone * DataSize, SAddr, &CShape, NULL, NULL, dtype );
    }
    /* T = V / ( N * H * W ) */
    tpu_bdc_fp_mul_C ( TAddr, VAddr, InvNHW, &CShape, NULL, NULL, dtype );
    /* T = T + eps */
    tpu_bdc_fp_add_C ( TAddr, TAddr, EPS, &CShape, NULL, NULL, dtype );
    /* V = RSQRT ( T ) */
    tpu_bdc_fp32_rsqrt ( VAddr, TAddr, &CShape );
    /* Move V as INVSTD from local to global memory */
    tpu_gdma_cpy_L2S ( batch_invstd_global_addr + CDone * DataSize, VAddr, &CShape, NULL, NULL, dtype );
    if ( weight_defined )
    {
      /* Move weight from global to local */
      tpu_gdma_cpy_S2L ( TAddr, weight_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
    }
    if ( bias_defined )
    {
      /* Move bias from global to local */
      tpu_gdma_cpy_S2L ( SAddr, bias_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Output */
    IOIndex = 0;
    NTodo = N, NDone = 0;
    int NDoneLast = 0, HDoneLast = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        if ( Split == true )
        {
          /* Move input from global to local */
          global_addr_t IGAddr = input_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
          tpu_gdma_cpy_S2L ( IOAddr[IOIndex], IGAddr, &Shape, NULL, &GlobalStride, dtype );
          if ( NDone > 0 || HDone > 0 )
          {
            tpu_parallel_end();
          }
          tpu_parallel_start();
          if ( NDone > 0 || HDone > 0 )
          {
            global_addr_t OGAddr = output_global_addr + ( NDoneLast * GlobalStride.n + CDone * GlobalStride.c + HDoneLast * GlobalStride.h ) * DataSize;
            tpu_gdma_cpy_L2S ( OGAddr, IOAddr[1 - IOIndex], &Shape, &GlobalStride, NULL, dtype );
          }
        }
        if ( Split == true )
        {
          /* O = I - M */
          tpu_bdc_fp_sub ( IOAddr[IOIndex], IOAddr[IOIndex], MAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          /* O = O * V */
          tpu_bdc_fp_mul ( IOAddr[IOIndex], IOAddr[IOIndex], VAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          if ( weight_defined )
          {
            /* O = O * T */
            tpu_bdc_fp_mul ( IOAddr[IOIndex], IOAddr[IOIndex], TAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          }
          if ( bias_defined )
          {
            /* O = O + S */
            tpu_bdc_fp_add ( IOAddr[IOIndex], IOAddr[IOIndex], SAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          }
        }
        else
        {
          /* O = I - M */
          tpu_bdc_fp_sub ( OAddr, IAddr, MAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          /* O = O * V */
          tpu_bdc_fp_mul ( OAddr, OAddr, VAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          if ( weight_defined )
          {
            /* O = O * T */
            tpu_bdc_fp_mul ( OAddr, OAddr, TAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          }
          if ( bias_defined )
          {
            /* O = O + S */
            tpu_bdc_fp_add ( OAddr, OAddr, SAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          }
        }
        /* Move output from local to global */
        if ( Split == true )
        {
          NDoneLast = NDone;
          HDoneLast = HDone;
        }
        else
        {
          global_addr_t OGAddr = output_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
          tpu_gdma_cpy_L2S ( OGAddr, OAddr, &Shape, &GlobalStride, NULL, dtype );
        }
        HTodo -= Shape.h;
        HDone += Shape.h;
        IOIndex = 1 - IOIndex;
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    if ( Split == true )
    {
      tpu_parallel_end();
      global_addr_t OGAddr = output_global_addr + ( NDoneLast * GlobalStride.n + CDone * GlobalStride.c + HDoneLast * GlobalStride.h ) * DataSize;
      tpu_gdma_cpy_L2S ( OGAddr, IOAddr[1 - IOIndex], &Shape, &GlobalStride, NULL, dtype );
    }
    CTodo -= Shape.c;
    CDone += Shape.c;
  }
}

static inline void nodechip_batchnorm2d_forward_training (
global_addr_t input_global_addr,
global_addr_t running_mean_global_addr,
global_addr_t running_var_global_addr,
global_addr_t weight_global_addr,
global_addr_t bias_global_addr,
global_addr_t updated_mean_global_addr,
global_addr_t updated_var_global_addr,
global_addr_t batch_mean_global_addr,
global_addr_t batch_invstd_global_addr,
global_addr_t output_global_addr,
dim4          shape,
float         momentum,
float         eps,
data_type_t   dtype )
{
  const bool weight_defined = weight_global_addr != 0;
  const bool bias_defined = bias_global_addr != 0;
  const bool running_mean_defined = running_mean_global_addr != 0;
  const bool running_var_defined = running_var_global_addr != 0;
  const int N = shape.n;
  const int C = shape.c;
  const int H = shape.h;
  const int W = shape.w;
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const int CPerNPU = DIV_UP ( C, NPU_NUM );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.f };
  const scalar_t InvNHW = { .f32 = 1.0 / ( ( double ) N * H * W ) };
  const scalar_t InvNHWMinusOne = { .f32 = 1.0 / ( ( double ) N * H * W - 1 ) };
  const scalar_t EPS = { .f32 = eps };
  const scalar_t Momentum = { .f32 = momentum };
  const scalar_t OneMinusMomentum = { .f32 = 1.f - momentum };
  const int DataSize = tpu_data_type_size ( dtype );
  /*
   *  I: [ NMax, CMax, HMax,     W ]
   *  O: [ NMax, CMax, HMax,     W ]
   *  P: [ NMax, CMax, HMax,     1 ]
   *  T: [ NMax, CMax,    1,     1 ]
   *  S: [ 1,    CMax,    1,  NMax ]
   *  M: [ 1,    CMax,    1,     1 ]
   *  V: [ 1,    CMax,    1,     1 ]
   */
  int HMax = H, NMax = N, CMax = CPerNPU * NPU_NUM;
  local_addr_t IAddr, OAddr, PAddr, TAddr, SAddr, MAddr, VAddr;
  bool Split = false;
  while ( true )
  {
    Split = NMax != N || HMax != H ;
    IAddr = 0;
    int ISize = tpu_aligned_feature_size ( HMax, W, dtype ) * DIV_UP ( CMax, NPU_NUM ) * NMax;
    if ( Split )
    {
      OAddr = IAddr;
    }
    else
    {
      OAddr = IAddr + ISize;
    }
    int OSize = ISize;
    PAddr = OAddr + OSize;
    int PSize = tpu_aligned_feature_size ( HMax, 1, dtype ) * DIV_UP ( CMax, NPU_NUM ) * NMax;
    TAddr = PAddr + PSize;
    int TSize = tpu_aligned_feature_size ( 1, 1, dtype ) * DIV_UP ( CMax, NPU_NUM ) * NMax;
    SAddr = TAddr + TSize;
    int SSize = tpu_aligned_feature_size ( 1, NMax, dtype ) * DIV_UP ( CMax, NPU_NUM );
    MAddr = SAddr + SSize;
    int MSize = tpu_aligned_feature_size ( 1,  1, dtype ) * DIV_UP ( CMax, NPU_NUM );
    VAddr = MAddr + MSize;
    int VSize = MSize;
    if ( ( int ) VAddr + VSize <= LOCAL_MEM_SIZE )
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
      else if ( NMax > 1 )
      {
        NMax /= 2;
        continue;
      }
      else if ( HMax > 1 )
      {
        HMax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  dim4 Shape = { .w = W };
  int CTodo = C, CDone = 0;
  while ( CTodo > 0 )
  {
    Shape.c = MIN ( CTodo, CMax );
    const dim4 CShape = { .n = 1, .c = Shape.c, .h = 1, .w = 1 };
    dim4 CStride;
    tpu_aligned_stride ( &CStride, 0, &CShape, dtype );
    dim4 CBcastStride = { .n = 0, .c = CStride.c, .h = 0, .w = 0 };
    /* Compute Mean */
    int NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = 1 };
      dim4 SShape = { .n = 1, .c = Shape.c, .h = 1, .w = Shape.n };
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        /* Move input from global to local */
        global_addr_t IGAddr = input_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
        tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, dtype );
        /* P = REDUCE_SUM ( I, [ 3 ] ) */
        dim2 KernelSizeDim3 = { .h = 1, .w = Shape.w };
        tpu_bdc_fp_avg_pool2d ( PAddr, IAddr, &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        /* T = REDUCE_SUM ( P, [ 2 ] ) */
        dim2 KernelSizeDim2 = { .h = Shape.h, .w = 1 };
        dim4 PShape = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
        tpu_bdc_fp_avg_pool2d ( TAddr, PAddr, &PShape, &KernelSizeDim2, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        dim4 TStride;
        tpu_aligned_stride ( &TStride, 0, &TShape, dtype );
        dim4 TTransStride = { .n = TStride.w, .c = TStride.c, .h = TStride.h, .w = TStride.n };
        if ( HDone == 0 )
        {
          /* S = NW-Transpose ( T ) */
          tpu_bdc_cpy ( SAddr, TAddr, &SShape, NULL, &TTransStride, dtype );
        }
        else
        {
          /* S = S + NW-Transpose ( T ) */
          tpu_bdc_fp_add ( SAddr, SAddr, TAddr, &SShape, NULL, NULL, &TTransStride, dtype );
        }
        HTodo -= Shape.h;
        HDone += Shape.h;
      }
      /* T or V = REDUCE_SUM ( S, [ 2, 3 ] ) */
      dim2 KernelSize = { .h = SShape.h, .w = SShape.w };
      tpu_bdc_fp_avg_pool2d ( NDone == 0 ? VAddr : TAddr, SAddr, &SShape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( NDone > 0 )
      {
        /* V = V + T */
        tpu_bdc_fp_add ( VAddr, VAddr, TAddr, &CShape, NULL, NULL, NULL, dtype );
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    /* M = V / ( N * H * W ) */
    tpu_bdc_fp_mul_C ( MAddr, VAddr, InvNHW, &CShape, NULL, NULL, dtype );
    /* Move M as MEAN from local to global memory */
    tpu_gdma_cpy_L2S ( batch_mean_global_addr + CDone * DataSize, MAddr, &CShape, NULL, NULL, dtype );
    if ( running_mean_defined )
    {
      /* Move running_mean from global to local memory */
      tpu_gdma_cpy_S2L ( TAddr, running_mean_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
      /* S = M * Momentum */
      tpu_bdc_fp_mul_C ( SAddr, MAddr, Momentum, &CShape, NULL, NULL, dtype );
      /* T = T * ( 1 - Momentum ) */
      tpu_bdc_fp_mul_C ( TAddr, TAddr, OneMinusMomentum, &CShape, NULL, NULL, dtype );
      /* T = T + S */
      tpu_bdc_fp_add ( TAddr, TAddr, SAddr, &CShape, NULL, NULL, NULL, dtype );
      /* Move T as UPDATED RUNNING MEAN from local to global memory */
      tpu_gdma_cpy_L2S ( updated_mean_global_addr + CDone * DataSize, TAddr, &CShape, NULL, NULL, dtype );
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Var */
    NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = 1 };
      dim4 SShape = { .n = 1, .c = Shape.c, .h = 1, .w = Shape.n };
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        if ( Split == true )
        {
          /* Move input from global to local */
          global_addr_t IGAddr = input_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
          tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, dtype );
        }
        /* O = I - M */
        tpu_bdc_fp_sub ( OAddr, IAddr, MAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        /* O = O * O */
        tpu_bdc_fp_mul ( OAddr, OAddr, OAddr, &Shape, NULL, NULL, NULL, dtype );
        /* P = REDUCE_SUM ( O, [ 3 ] ) */
        dim2 KernelSizeDim3 = { .h = 1, .w = Shape.w };
        tpu_bdc_fp_avg_pool2d ( PAddr, OAddr, &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        /* T = REDUCE_SUM ( P, [ 2 ] ) */
        dim2 KernelSizeDim2 = { .h = Shape.h, .w = 1 };
        dim4 PShape = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
        tpu_bdc_fp_avg_pool2d ( TAddr, PAddr, &PShape, &KernelSizeDim2, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        dim4 TStride;
        tpu_aligned_stride ( &TStride, 0, &TShape, dtype );
        dim4 TTransStride = { .n = TStride.w, .c = TStride.c, .h = TStride.h, .w = TStride.n };
        if ( HDone == 0 )
        {
          /* S = NW-Transpose ( T ) */
          tpu_bdc_cpy ( SAddr, TAddr, &SShape, NULL, &TTransStride, dtype );
        }
        else
        {
          /* S = S + NW-Transpose ( T ) */
          tpu_bdc_fp_add ( SAddr, SAddr, TAddr, &SShape, NULL, NULL, &TTransStride, dtype );
        }
        HTodo -= Shape.h;
        HDone += Shape.h;
      }
      /* T or V = REDUCE_SUM ( S, [ 2, 3 ] ) */
      dim2 KernelSize = { .h = SShape.h, .w = SShape.w };
      tpu_bdc_fp_avg_pool2d ( NDone == 0 ? VAddr : TAddr, SAddr, &SShape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( NDone > 0 )
      {
        /* V = V + T */
        tpu_bdc_fp_add ( VAddr, VAddr, TAddr, &CShape, NULL, NULL, NULL, dtype );
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    if ( running_var_defined )
    {
      /* Move running_var from global to local memory */
      tpu_gdma_cpy_S2L ( PAddr, running_var_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
      /* T = V / ( N * H * W - 1 ) */
      tpu_bdc_fp_mul_C ( TAddr, VAddr, InvNHWMinusOne, &CShape, NULL, NULL, dtype );
      /* S = T * Momentum */
      tpu_bdc_fp_mul_C ( SAddr, TAddr, Momentum, &CShape, NULL, NULL, dtype );
      /* P = P * ( 1 - Momentum ) */
      tpu_bdc_fp_mul_C ( PAddr, PAddr, OneMinusMomentum, &CShape, NULL, NULL, dtype );
      /* S = S + P */
      tpu_bdc_fp_add ( SAddr, SAddr, PAddr, &CShape, NULL, NULL, NULL, dtype );
      /* Move T as UPDATED RUNNING VAR from local to global memory */
      tpu_gdma_cpy_L2S ( updated_var_global_addr + CDone * DataSize, SAddr, &CShape, NULL, NULL, dtype );
    }
    /* T = V / ( N * H * W ) */
    tpu_bdc_fp_mul_C ( TAddr, VAddr, InvNHW, &CShape, NULL, NULL, dtype );
    /* T = T + eps */
    tpu_bdc_fp_add_C ( TAddr, TAddr, EPS, &CShape, NULL, NULL, dtype );
    /* V = RSQRT ( T ) */
    tpu_bdc_fp32_rsqrt ( VAddr, TAddr, &CShape );
    /* Move V as INVSTD from local to global memory */
    tpu_gdma_cpy_L2S ( batch_invstd_global_addr + CDone * DataSize, VAddr, &CShape, NULL, NULL, dtype );
    if ( weight_defined )
    {
      /* Move weight from global to local */
      tpu_gdma_cpy_S2L ( TAddr, weight_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
    }
    if ( bias_defined )
    {
      /* Move bias from global to local */
      tpu_gdma_cpy_S2L ( SAddr, bias_global_addr + CDone * DataSize, &CShape, NULL, NULL, dtype );
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /* Compute Output */
    NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        if ( Split == true )
        {
          /* Move input from global to local */
          global_addr_t IGAddr = input_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
          tpu_gdma_cpy_S2L ( IAddr, IGAddr, &Shape, NULL, &GlobalStride, dtype );
        }
        /* O = I - M */
        tpu_bdc_fp_sub ( OAddr, IAddr, MAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        /* O = O * V */
        tpu_bdc_fp_mul ( OAddr, OAddr, VAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        if ( weight_defined )
        {
          /* O = O * T */
          tpu_bdc_fp_mul ( OAddr, OAddr, TAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        }
        if ( bias_defined )
        {
          /* O = O + S */
          tpu_bdc_fp_add ( OAddr, OAddr, SAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        }
        /* Move output from local to global */
        global_addr_t OGAddr = output_global_addr + ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
        tpu_gdma_cpy_L2S ( OGAddr, OAddr, &Shape, &GlobalStride, NULL, dtype );
        HTodo -= Shape.h;
        HDone += Shape.h;
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    CTodo -= Shape.c;
    CDone += Shape.c;
  }
}

void tpu_kernel_api_batchnorm_forward_v2 ( const void * args )
{
  sg_api_batchnorm_forward_t *api = ( sg_api_batchnorm_forward_t * ) args;
  dim4 shape = { api->shape[0], api->shape[1], api->shape[2], api->shape[3] };
  TPUKERNEL_ASSERT ( api->dtype == SG_DTYPE_FP32 );
  tpu_initialize();
  nodechip_batchnorm2d_forward_training_parallel (
  api->input_global_addr,
  api->running_mean_global_addr,
  api->running_var_global_addr,
  api->weight_global_addr,
  api->bias_global_addr,
  api->updated_mean_global_addr,
  api->updated_var_global_addr,
  api->batch_mean_global_addr,
  api->batch_invstd_global_addr,
  api->output_global_addr,
  shape,
  api->momentum,
  api->eps,
  DT_FP32 );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_batchnorm_forward_v2 );
