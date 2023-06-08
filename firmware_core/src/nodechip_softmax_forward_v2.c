#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

void nodechip_softmax_forward_2DR1 (
global_addr_t IGAddr,
global_addr_t OGAddr,
int Row, // row number
int Column, // column number
data_type_t DType ) // DT_FP32 or DT_FP16
{
  const int DSize = tpu_data_type_size ( DType );
  const int Tile = tpu_eu_num ( DType );
  const dim4 TotalShape = { .n = 1, .c = Row, .h = Column, .w = 1 };
  dim4 IGStride, OGStride;
  tpu_continuous_stride ( &IGStride, &TotalShape );
  tpu_continuous_stride ( &OGStride, &TotalShape );
  const dim2 StrideOne = { .h = 1, .w = 1 };
  const dim2 DilationOne = { .h = 1, .w = 1 };
  const padding_t ZeroPad = { .top = 0, .left = 0, .bottom = 0, .right = 0 };
  const scalar_t Zero = { .u32 = 0 };
  const scalar_t OneFP32 = { .f32 = 1.f };
  local_addr_t IAddr, RTAddr, RMAddrs[2], RSAddrs[2];
  local_addr_t IFP32Addr, W0Addr, W1Addr;
  int CMax = Row, HMax = Column;
  local_addr_t ETAddr = 0;
  local_addr_t ECAddr = ETAddr + tpu_aligned_feature_size ( 1, 192, DT_FP32 );
  IAddr = ECAddr + tpu_aligned_feature_size ( 1, 32, DT_FP32 );
  while ( true )
  {
    int Size = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( HMax, Tile ), Tile, DType );
    int SizeFP32 = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( DIV_UP ( HMax, Tile ), Tile, DT_FP32 );
    int TSizeFP32 = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, Tile, DT_FP32 );
    int MSize = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DType );
    int SSizeFP32 = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    int Next = IAddr + Size;
    if ( DType == DT_FP16 )
    {
      IFP32Addr = Next; Next = IFP32Addr + SizeFP32;
    }
    else
    {
      IFP32Addr = IAddr;
    }
    RTAddr = Next; Next = RTAddr + TSizeFP32;
    RMAddrs[0] = Next; Next = RMAddrs[0] + MSize;
    if ( HMax != Column )
    {
      RMAddrs[1] = Next; Next = RMAddrs[1] + MSize;
    }
    else
    {
      RMAddrs[1] = RMAddrs[0];
    }
    RSAddrs[0] = Next; Next = RSAddrs[0] + SSizeFP32;
    if ( HMax != Column )
    {
      RSAddrs[1] = Next; Next = RSAddrs[1] + SSizeFP32;
    }
    else
    {
      RSAddrs[1] = RSAddrs[0];
    }
    W0Addr = Next; Next = W0Addr + SizeFP32;
    W1Addr = Next; Next = W1Addr + SizeFP32;
    if ( Next <= LOCAL_MEM_SIZE )
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
      else if ( HMax > 1 )
      {
        HMax -= Tile;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  tpu_bdc_load_fp32_exp_table ( ETAddr );
  tpu_bdc_load_fp32_exp_coeff ( ECAddr );
  dim4 Shape = { .n = 1, .w = 1 };
  int CTodo = Row;
  int CDone = 0;
  while ( CTodo != 0 )
  {
    Shape.c = MIN ( CMax, CTodo );
    dim4 RShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = 1 };
    int HTodo = Column, HDone = 0;
    int Index = 0; // Ping-pong buffer switching index
    /*
     * Find the max value in row
     */
    while ( HTodo != 0 )
    {
      Shape.h = MIN ( HMax, HTodo );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = DIV_UP ( Shape.h, Tile ), .w = Tile };
      dim4 TStride; tpu_aligned_stride ( &TStride, 0, &TShape, DType );
      // Move input from global memory to local memory
      dim4 Stride = { .n = TStride.n, .c = TStride.c, .h = 1, .w = 1 };
      tpu_gdma_cpy_S2L ( IAddr, IGAddr + ( CDone * IGStride.c + HDone * IGStride.h ) * DSize, &Shape, &Stride, &IGStride, DType );
      // set input tail
      if ( Shape.h % Tile != 0 )
      {
        scalar_t C = { .u32 = DType == DT_FP16 ? 0xfbff : 0xff7fffff };
        dim4 SShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = Tile - ( Shape.h % Tile ) };
        tpu_bdc_set_C ( IAddr + Shape.h * DSize, C, &SShape, &TStride, DType );
      }
      // [ 1, row, DIV_UP ( column, Tile ), Tile ] -> [ 1, row, 1, Tile ] -> [ 1, row, 1, 1 ]
      dim2 Kernel = { .h = TShape.h, .w = 1 };
      tpu_bdc_fp_max_pool2d ( RTAddr, IAddr, &TShape, &Kernel, &ZeroPad, &StrideOne, &DilationOne, DType, Zero );
      TShape.h = 1; Kernel.h = 1; Kernel.w = TShape.w;
      tpu_bdc_fp_max_pool2d ( RMAddrs[Index], RTAddr, &TShape, &Kernel, &ZeroPad, &StrideOne, &DilationOne, DType, Zero );
      if ( HDone > 0 )
      {
        tpu_bdc_max ( RMAddrs[Index], RMAddrs[Index], RMAddrs[1 - Index], &RShape, NULL, NULL, NULL, DType );
      }
      HTodo -= Shape.h;
      HDone += Shape.h;
      Index = 1 - Index;
    }
    local_addr_t MAddr = RMAddrs[1 - Index];
    HTodo = Column; HDone = 0; Index = 0;
    /*
     * input = input - input_max
     * input = exp ( input )
     * input_sum = sum_row ( input )
     */
    while ( HTodo != 0 )
    {
      Shape.h = MIN ( HMax, HTodo );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = DIV_UP ( Shape.h, Tile ), .w = Tile };
      if ( HMax != Column )
      {
        // Move input from global memory to local memory
        dim4 TStride; tpu_aligned_stride ( &TStride, 0, &TShape, DType );
        dim4 Stride = { .n = TStride.n, .c = TStride.c, .h = 1, .w = 1 };
        tpu_gdma_cpy_S2L ( IAddr, IGAddr + ( CDone * IGStride.c + HDone * IGStride.h ) * DSize, &Shape, &Stride, &IGStride, DType );
      }
      // input = input - input_max
      dim4 MStride; tpu_aligned_stride ( &MStride, 0, &RShape, DType );
      dim4 MBStride = { .n = MStride.n, .c = MStride.c, 0, 0 };
      tpu_bdc_fp_sub ( IAddr, IAddr, MAddr, &TShape, NULL, NULL, &MBStride, DType );
      // input FP16 -> FP32
      if ( DType == DT_FP16 )
      {
        tpu_bdc_cast ( IFP32Addr, IAddr, &TShape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
      }
      // input = exp ( input )
      tpu_bdc_fp32_exp ( IFP32Addr, IFP32Addr, W0Addr, W1Addr, ECAddr, ETAddr, &TShape );
      // set input tail
      if ( Shape.h % Tile != 0 )
      {
        dim4 TStride; tpu_aligned_stride ( &TStride, 0, &TShape, DT_FP32 );
        dim4 SShape = { .n = Shape.n, .c = Shape.c, .h = 1, .w = Tile - ( Shape.h % Tile ) };
        tpu_bdc_set_C ( IFP32Addr + Shape.h * 4, Zero, &SShape, &TStride, DT_FP32 );
      }
      // [ 1, row, DIV_UP ( column, Tile ), Tile ] -> [ 1, row, 1, Tile ] -> [ 1, row, 1, 1 ]
      dim2 Kernel = { .h = TShape.h, .w = 1 };
      tpu_bdc_fp_avg_pool2d ( RTAddr, IFP32Addr, &TShape, &Kernel, &ZeroPad, &StrideOne, &DilationOne, DT_FP32, OneFP32 );
      TShape.h = 1; Kernel.h = 1; Kernel.w = TShape.w;
      tpu_bdc_fp_avg_pool2d ( RSAddrs[Index], RTAddr, &TShape, &Kernel, &ZeroPad, &StrideOne, &DilationOne, DT_FP32, OneFP32 );
      if ( HDone > 0 )
      {
        tpu_bdc_fp_add ( RSAddrs[Index], RSAddrs[Index], RSAddrs[1 - Index], &RShape, NULL, NULL, NULL, DT_FP32 );
      }
      HTodo -= Shape.h;
      HDone += Shape.h;
      Index = 1 - Index;
    }
    local_addr_t SAddr = RSAddrs[1 - Index];
    // input_sum = 1 / input_sum
    tpu_bdc_fp32_reciprocal ( SAddr, SAddr, &RShape, NULL, NULL );
    HTodo = Column; HDone = 0;
    /*
     * input = input * input_sum
     */
    while ( HTodo != 0 )
    {
      Shape.h = MIN ( HMax, HTodo );
      dim4 TShape = { .n = Shape.n, .c = Shape.c, .h = DIV_UP ( Shape.h, Tile ), .w = Tile };
      if ( HMax != Column )
      {
        // Move input from global memory to local memory
        dim4 TStride; tpu_aligned_stride ( &TStride, 0, &TShape, DType );
        dim4 Stride = { .n = TStride.n, .c = TStride.c, .h = 1, .w = 1 };
        tpu_gdma_cpy_S2L ( IAddr, IGAddr + ( CDone * IGStride.c + HDone * IGStride.h ) * DSize, &Shape, &Stride, &IGStride, DType );
        // input = input - input_max
        dim4 MStride; tpu_aligned_stride ( &MStride, 0, &RShape, DType );
        dim4 MBStride = { .n = MStride.n, .c = MStride.c, 0, 0 };
        tpu_bdc_fp_sub ( IAddr, IAddr, MAddr, &TShape, NULL, NULL, &MBStride, DType );
        // input FP16 -> FP32
        if ( DType == DT_FP16 )
        {
          tpu_bdc_cast ( IFP32Addr, IAddr, &TShape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
        }
        // input = exp ( input )
        tpu_bdc_fp32_exp ( IFP32Addr, IFP32Addr, W0Addr, W1Addr, ECAddr, ETAddr, &TShape );
      }
      // input = input * input_sum
      dim4 SStride; tpu_aligned_stride ( &SStride, 0, &RShape, DT_FP32 );
      dim4 SBStride = { .n = SStride.n, .c = SStride.c, 0, 0 };
      tpu_bdc_fp_mul ( IFP32Addr, IFP32Addr, SAddr, &TShape, NULL, NULL, &SBStride, DT_FP32 );
      // input FP32 -> FP16 (inplace)
      if ( DType == DT_FP16 )
      {
        tpu_bdc_cast ( IFP32Addr, IFP32Addr, &TShape, NULL, NULL, DT_FP16, DT_FP32, RM_HALF_TO_EVEN );
      }
      // Move output from local memory to global memory
      dim4 TStride; tpu_aligned_stride ( &TStride, 0, &TShape, DType );
      dim4 Stride = { .n = TStride.n, .c = TStride.c, .h = 1, .w = 1 };
      tpu_gdma_cpy_L2S ( OGAddr + ( CDone * OGStride.c + HDone * OGStride.h ) * DSize, IFP32Addr, &Shape, &OGStride, &Stride, DType );
      HTodo -= Shape.h;
      HDone += Shape.h;
    }
    CTodo -= Shape.c;
    CDone += Shape.c;
  }
}
