#include "sg_api_struct.h"
#include "tpu_kernel.h"

// layernorm backward only support dim3(w_dim) for now
static inline void nodechip_layernorm_backward_dim3_parallel (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t mean_global_addr,
global_addr_t invstd_global_addr,
global_addr_t grad_input_global_addr,
global_addr_t grad_weight_global_addr,
global_addr_t grad_bias_global_addr,
dim4 shape,
data_type_t dtype,
int grad_input_enable,
int grad_weight_enable,
int grad_bias_enable )
{
  const int N = shape.n;
  const int C = shape.c;
  const int H = shape.h;
  const int W = shape.w;
  TPUKERNEL_ASSERT ( H == 1 );
  TPUKERNEL_ASSERT ( grad_input_enable );
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 WeightShape = { .n = 1, .c = 1, .h = 1, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const int CPerNPU = DIV_UP ( C, NPU_NUM );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.0 };
  // const scalar_t InvNC = { .f32 = 1.0 / ( ( double ) N * C ) };
  const scalar_t InvW = { .f32 = 1.0 / ( ( double ) W ) };
  const int DataSize = tpu_data_type_size ( dtype );
  /*
  *  grad_output    : [ NMax, CMax,    1,     W ]
  *  input          : [ NMax, CMax,    1,     W ]
  *  prod           : [ NMax, CMax,    1,     W ]
  *  mean           : [ NMax, CMax,    1,     1 ]
  *  invstd         : [ NMax, CMax,    1,     1 ]
  *  weight         : [ 1,     NPU,    1,     W ]
  *  trans_tmp1     : [ NMax,    W,    1,  CMax ]
  *  trans_tmp2     : [ 1,       W, NMax,  CMax ]
  *  reduce_tmp     : [ 1,       W,    1,     1 ]
  *  sum_tmp        : [ 1,       1,    1,     W ]
  *  grad_bias      : [ 1,       1,    1,     W ]
  *  grad_weight    : [ 1,       1,    1,     W ]
  */
  int NMax = N, CMax = CPerNPU * NPU_NUM;
  local_addr_t grad_inputAddr, grad_outputAddr;
  local_addr_t inputAddr, input_normAddr, prodAddr;
  local_addr_t meanAddr, invstdAddr, weightAddr, bufferAddr;
  local_addr_t tmp1Addr, tmp2Addr, reducetmpAddr, sumtmpAddr;
  local_addr_t grad_biasAddr, grad_weightAddr;
  bool Split = false;
  while ( true )
  {
    Split = NMax != N || CMax != C ;
    grad_outputAddr = 0;
    int grad_outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, dtype );
    inputAddr = grad_outputAddr + grad_outputSize;
    int inputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, dtype );
    prodAddr = inputAddr + inputSize;
    int prodSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, dtype );
    meanAddr = prodAddr + prodSize;
    int meanSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    invstdAddr = meanAddr + meanSize;
    int invstdSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    weightAddr = invstdAddr + invstdSize;
    int weightSize = tpu_aligned_feature_size ( 1, W, dtype );
    tmp1Addr = weightAddr + weightSize;
    int tmp1Size = NMax * DIV_UP ( W, NPU_NUM ) * tpu_aligned_feature_size ( 1, CMax, dtype );
    tmp2Addr = tmp1Addr + tmp1Size;
    int tmp2Size = DIV_UP ( W, NPU_NUM ) * tpu_aligned_feature_size ( NMax, CMax, dtype );
    reducetmpAddr = tmp2Addr + tmp2Size;
    int reducetmpSize = DIV_UP ( W, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    sumtmpAddr = reducetmpAddr + reducetmpSize;
    int sumtmpSize = tpu_aligned_feature_size ( 1, W, dtype );
    grad_biasAddr = sumtmpAddr + sumtmpSize;
    int grad_biasSize = tpu_aligned_feature_size ( 1, W, dtype );
    grad_weightAddr = grad_biasAddr + grad_biasSize;
    int grad_weightSize = tpu_aligned_feature_size ( 1, W, dtype );
    input_normAddr = inputAddr;
    grad_inputAddr = grad_outputAddr;
    bufferAddr = meanAddr;
    if ( ( int ) grad_weightAddr + grad_weightSize <= LOCAL_MEM_SIZE ) { break; }
    else
    {
      if ( NMax > 1 )
      {
        NMax /= 2;
        continue;
      }
      if ( CMax > NPU_NUM )
      {
        CMax -= NPU_NUM;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  dim4 Shape = { .h = 1, .w = W };
  int NTodo = N, NDone = 0;
  while ( NTodo > 0 )
  {
    Shape.n = MIN ( NTodo, NMax );
    int CTodo = C, CDone = 0;
    while ( CTodo > 0 )
    {
      Shape.c = MIN ( CTodo, CMax );
      dim4 ParamShape   = { .n = Shape.n, .c = Shape.c, .h = 1,       .w = 1 };
      dim4 TransShape   = { .n = Shape.n, .c = Shape.w, .h = 1,       .w = Shape.c };
      dim4 PoolingShape = { .n = 1,       .c = Shape.w, .h = Shape.n, .w = Shape.c };
      int TotalDone = ( NDone * GlobalStride.n + CDone * GlobalStride.c ) * DataSize;
      global_addr_t grad_inputGAddr = grad_input_global_addr + TotalDone;
      global_addr_t grad_outputGAddr = grad_output_global_addr + TotalDone;
      global_addr_t inputGAddr = input_global_addr + TotalDone;
      global_addr_t meanGAddr = mean_global_addr + ( NDone * C + CDone ) * DataSize;
      global_addr_t invstdGAddr = invstd_global_addr + ( NDone * C + CDone ) * DataSize;
      global_addr_t weightGAddr = weight_global_addr;
      /* move input & grad_output */
      tpu_gdma_cpy_S2L ( grad_outputAddr, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
      tpu_gdma_cpy_S2L ( inputAddr, inputGAddr, &Shape, NULL, &GlobalStride, dtype );
      /* move mean & invstd */
      tpu_gdma_cpy_S2L ( meanAddr, meanGAddr, &ParamShape, NULL, NULL, dtype );
      tpu_gdma_cpy_S2L ( invstdAddr, invstdGAddr, &ParamShape, NULL, NULL, dtype );
      /* move and broadcast weight to all lane */
      if ( !NDone && !CDone )
      {
        tpu_gdma_cpy_S2L ( weightAddr, weightGAddr, &WeightShape, NULL, NULL, dtype );
        WeightShape.c = NPU_NUM;
        tpu_bdc_npu_bcast ( weightAddr, weightAddr, &WeightShape, dtype );
        WeightShape.c = 1;
      }
      /* compute normed input & product with grad output */
      dim4 WBcastStride;
      tpu_aligned_stride ( &WBcastStride, 0, &ParamShape, dtype );
      WBcastStride.w = 0;
      tpu_bdc_fp_sub ( input_normAddr, inputAddr, meanAddr, &Shape, NULL, NULL, &WBcastStride, dtype );
      tpu_bdc_fp_mul ( input_normAddr, input_normAddr, invstdAddr, &Shape, NULL, NULL, &WBcastStride, dtype );
      tpu_bdc_fp_mul ( prodAddr, input_normAddr, grad_outputAddr, &Shape, NULL, NULL, NULL, dtype );
      /* pooling prod & grad output in nc-dim */
      dim2 KernelSizeNCdim = { .h = Shape.n, .w = Shape.c };
      dim4 TransStride;
      tpu_aligned_stride ( &TransStride, 0, &TransShape, dtype );
      dim4 TTransStride = { .n = TransStride.h, .c = TransStride.c, .h = TransStride.n, .w = TransStride.w };
      tpu_bdc_cw_trans ( tmp1Addr, grad_outputAddr, &TransShape, dtype );
      tpu_bdc_cpy ( tmp2Addr, tmp1Addr, &PoolingShape, NULL, &TTransStride, dtype );
      tpu_bdc_fp_avg_pool2d ( reducetmpAddr, tmp2Addr, &PoolingShape, &KernelSizeNCdim, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( !Split || ( !NDone && !CDone ) )
      {
        tpu_bdc_cw_trans ( grad_biasAddr, reducetmpAddr, &WeightShape, dtype );
      }
      else
      {
        tpu_bdc_cw_trans ( sumtmpAddr, reducetmpAddr, &WeightShape, dtype );
        tpu_bdc_fp_add ( grad_biasAddr, grad_biasAddr, sumtmpAddr, &WeightShape, NULL, NULL, NULL, dtype );
      }
      tpu_bdc_cw_trans ( tmp1Addr, prodAddr, &TransShape, dtype );
      tpu_bdc_cpy ( tmp2Addr, tmp1Addr, &PoolingShape, NULL, &TTransStride, dtype );
      tpu_bdc_fp_avg_pool2d ( reducetmpAddr, tmp2Addr, &PoolingShape, &KernelSizeNCdim, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( !Split || ( !NDone && !CDone ) )
      {
        tpu_bdc_cw_trans ( grad_weightAddr, reducetmpAddr, &WeightShape, dtype );
      }
      else
      {
        tpu_bdc_cw_trans ( sumtmpAddr, reducetmpAddr, &WeightShape, dtype );
        tpu_bdc_fp_add ( grad_weightAddr, grad_weightAddr, sumtmpAddr, &WeightShape, NULL, NULL, NULL, dtype );
      }
      /* compute prod = weight * grad output * normed input */
      dim4 NCBcastStride;
      tpu_aligned_stride ( &NCBcastStride, 0, &WeightShape, dtype );
      NCBcastStride.n = 0;
      NCBcastStride.c = 0;
      tpu_bdc_fp_mul ( grad_outputAddr, grad_outputAddr, weightAddr, &Shape, NULL, NULL, &NCBcastStride, dtype );
      tpu_bdc_fp_mul ( prodAddr, prodAddr, weightAddr, &Shape, NULL, NULL, &NCBcastStride, dtype );
      /* pooling prod & grad output in w-dim */
      dim2 KernelSizeWdim = { .h = 1, .w = Shape.w };
      tpu_bdc_fp_avg_pool2d ( bufferAddr, prodAddr, &Shape, &KernelSizeWdim, &ZeroPadding, &OneStride, &OneDilation, dtype, InvW );
      tpu_bdc_fp_mul ( input_normAddr, input_normAddr, bufferAddr, &Shape, NULL, NULL, &WBcastStride, dtype );
      tpu_bdc_fp_avg_pool2d ( bufferAddr, grad_outputAddr, &Shape, &KernelSizeWdim, &ZeroPadding, &OneStride, &OneDilation, dtype, InvW );
      tpu_bdc_fp_add ( input_normAddr, input_normAddr, bufferAddr, &Shape, NULL, NULL, &WBcastStride, dtype );
      /* compute and move grad input */
      tpu_bdc_fp_sub ( grad_inputAddr, grad_outputAddr, inputAddr, &Shape, NULL, NULL, NULL, dtype );
      tpu_bdc_fp_mul ( grad_inputAddr, grad_inputAddr, invstdAddr, &Shape, NULL, NULL, &WBcastStride, dtype );
      tpu_gdma_cpy_L2S ( grad_inputGAddr, grad_inputAddr, &Shape, &GlobalStride, NULL, dtype );
      CTodo -= Shape.c;
      CDone += Shape.c;
    }
    NTodo -= Shape.n;
    NDone += Shape.n;
  }
  if ( grad_bias_enable ) tpu_gdma_cpy_L2S ( grad_bias_global_addr, grad_biasAddr, &WeightShape, NULL, NULL, dtype );
  if ( grad_weight_enable ) tpu_gdma_cpy_L2S ( grad_weight_global_addr, grad_weightAddr, &WeightShape, NULL, NULL, dtype );
}

static inline void nodechip_layernorm_backward_dim3 (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t mean_global_addr,
global_addr_t invstd_global_addr,
global_addr_t grad_input_global_addr,
global_addr_t grad_weight_global_addr,
global_addr_t grad_bias_global_addr,
dim4 shape,
data_type_t dtype,
int grad_input_enable,
int grad_weight_enable,
int grad_bias_enable )
{
  const int N = shape.n;
  const int C = shape.c;
  const int H = shape.h;
  const int W = shape.w;
  TPUKERNEL_ASSERT ( H == 1 );
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 WeightShape = { .n = 1, .c = 1, .h = 1, .w = W };
  dim4 WeightBCastShape = { .n = 1, .c = NPU_NUM, .h = 1, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const int CPerNPU = DIV_UP ( C, NPU_NUM );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.0 };
  const scalar_t InvW = { .f32 = 1.0 / ( ( double ) W ) };
  const int DataSize = tpu_data_type_size ( dtype );
  /*
  *  grad_output    : [ NMax, CMax,    1,     W ]
  *  input          : [ NMax, CMax,    1,     W ]
  *  prod           : [ NMax, CMax,    1,     W ]
  *  mean           : [ NMax, CMax,    1,     1 ]
  *  invstd         : [ NMax, CMax,    1,     1 ]
  *  weight         : [ 1,     NPU,    1,     W ]
  *  trans_tmp1     : [ NMax,    W,    1,  CMax ]
  *  trans_tmp2     : [ 1,       W, NMax,  CMax ]
  *  reduce_tmp     : [ 1,       W,    1,     1 ]
  *  sum_tmp        : [ 1,       1,    1,     W ]
  *  grad_bias      : [ 1,       1,    1,     W ]
  *  grad_weight    : [ 1,       1,    1,     W ]
  */
  int NMax = N, CMax = CPerNPU * NPU_NUM;
  local_addr_t grad_inputAddr, grad_outputAddr;
  local_addr_t inputAddr, input_normAddr, prodAddr;
  local_addr_t meanAddr, invstdAddr, weightAddr, bufferAddr;
  local_addr_t tmp1Addr, tmp2Addr, reducetmpAddr, sumtmpAddr;
  local_addr_t grad_biasAddr, grad_weightAddr;
  local_addr_t grad_outputAddr_fp16, inputAddr_fp16, meanAddr_fp16, invstdAddr_fp16, weightAddr_fp16;
  int grad_outputSize_fp16, inputSize_fp16, meanSize_fp16, invstdSize_fp16, weightSize_fp16;
  bool Split = false;
  while ( true )
  {
    Split = NMax != N || CMax != C ;
    grad_outputAddr = 0;
    if ( dtype != DT_FP32 )
    {
      grad_outputAddr_fp16 = 0;
      grad_outputSize_fp16 = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, dtype );
      inputAddr_fp16 = grad_outputAddr_fp16 + grad_outputSize_fp16;
      inputSize_fp16 = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, dtype );
      meanAddr_fp16 = inputAddr_fp16 + inputSize_fp16;
      meanSize_fp16 = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
      invstdAddr_fp16 = meanAddr_fp16 + meanSize_fp16;
      invstdSize_fp16 = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
      weightAddr_fp16 = invstdAddr_fp16 + invstdSize_fp16;
      weightSize_fp16 = tpu_aligned_feature_size ( 1, W, dtype );
      grad_outputAddr = weightAddr_fp16 + weightSize_fp16;
    }
    int grad_outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, DT_FP32 );
    inputAddr = grad_outputAddr + grad_outputSize;
    int inputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, DT_FP32 );
    prodAddr = inputAddr + inputSize;
    int prodSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, W, DT_FP32 );
    meanAddr = prodAddr + prodSize;
    int meanSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    invstdAddr = meanAddr + meanSize;
    int invstdSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    weightAddr = invstdAddr + invstdSize;
    int weightSize = tpu_aligned_feature_size ( 1, W, DT_FP32 );
    tmp1Addr = weightAddr + weightSize;
    int tmp1Size = NMax * DIV_UP ( W, NPU_NUM ) * tpu_aligned_feature_size ( 1, CMax, DT_FP32 );
    tmp2Addr = tmp1Addr + tmp1Size;
    int tmp2Size = DIV_UP ( W, NPU_NUM ) * tpu_aligned_feature_size ( NMax, CMax, DT_FP32 );
    reducetmpAddr = tmp2Addr + tmp2Size;
    int reducetmpSize = DIV_UP ( W, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, DT_FP32 );
    sumtmpAddr = reducetmpAddr + reducetmpSize;
    int sumtmpSize = tpu_aligned_feature_size ( 1, W, DT_FP32 );
    grad_biasAddr = sumtmpAddr + sumtmpSize;
    int grad_biasSize = tpu_aligned_feature_size ( 1, W, DT_FP32 );
    grad_weightAddr = grad_biasAddr + grad_biasSize;
    int grad_weightSize = tpu_aligned_feature_size ( 1, W, DT_FP32 );
    input_normAddr = inputAddr;
    grad_inputAddr = grad_outputAddr;
    bufferAddr = meanAddr;
    if ( ( int ) grad_weightAddr + grad_weightSize <= LOCAL_MEM_SIZE ) { break; }
    else
    {
      if ( NMax > 1 )
      {
        NMax /= 2;
        continue;
      }
      if ( CMax > NPU_NUM )
      {
        CMax -= NPU_NUM;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  dim4 Shape = { .h = 1, .w = W };
  int NTodo = N, NDone = 0;
  while ( NTodo > 0 )
  {
    Shape.n = MIN ( NTodo, NMax );
    int CTodo = C, CDone = 0;
    while ( CTodo > 0 )
    {
      Shape.c = MIN ( CTodo, CMax );
      dim4 ParamShape   = { .n = Shape.n, .c = Shape.c, .h = 1,       .w = 1 };
      dim4 TransShape   = { .n = Shape.n, .c = Shape.w, .h = 1,       .w = Shape.c };
      dim4 PoolingShape = { .n = 1,       .c = Shape.w, .h = Shape.n, .w = Shape.c };
      int TotalDone = ( NDone * GlobalStride.n + CDone * GlobalStride.c ) * DataSize;
      global_addr_t grad_inputGAddr = grad_input_global_addr + TotalDone;
      global_addr_t grad_outputGAddr = grad_output_global_addr + TotalDone;
      global_addr_t inputGAddr = input_global_addr + TotalDone;
      global_addr_t meanGAddr = mean_global_addr + ( NDone * C + CDone ) * DataSize;
      global_addr_t invstdGAddr = invstd_global_addr + ( NDone * C + CDone ) * DataSize;
      global_addr_t weightGAddr = weight_global_addr;
      if ( dtype != DT_FP32 )
      {
        tpu_gdma_cpy_S2L ( grad_outputAddr_fp16, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
        tpu_gdma_cpy_S2L ( inputAddr_fp16, inputGAddr, &Shape, NULL, &GlobalStride, dtype );
        tpu_gdma_cpy_S2L ( meanAddr_fp16, meanGAddr, &ParamShape, NULL, NULL, dtype );
        tpu_gdma_cpy_S2L ( invstdAddr_fp16, invstdGAddr, &ParamShape, NULL, NULL, dtype );
        if ( !NDone && !CDone )
        {
          tpu_gdma_cpy_S2L ( weightAddr_fp16, weightGAddr, &WeightShape, NULL, NULL, dtype );
          tpu_bdc_npu_bcast ( weightAddr_fp16, weightAddr_fp16, &WeightBCastShape, dtype );
        }
        tpu_bdc_cast ( grad_outputAddr, grad_outputAddr_fp16, &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_cast ( inputAddr, inputAddr_fp16, &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_cast ( meanAddr, meanAddr_fp16, &ParamShape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_cast ( invstdAddr, invstdAddr_fp16, &ParamShape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        tpu_bdc_cast ( weightAddr, weightAddr_fp16, &WeightBCastShape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
      }
      else if ( dtype == DT_FP32 )
      {
        /* move input & grad_output */
        tpu_gdma_cpy_S2L ( grad_outputAddr, grad_outputGAddr, &Shape, NULL, &GlobalStride, DT_FP32 );
        tpu_gdma_cpy_S2L ( inputAddr, inputGAddr, &Shape, NULL, &GlobalStride, DT_FP32 );
        /* move mean & invstd */
        tpu_gdma_cpy_S2L ( meanAddr, meanGAddr, &ParamShape, NULL, NULL, DT_FP32 );
        tpu_gdma_cpy_S2L ( invstdAddr, invstdGAddr, &ParamShape, NULL, NULL, DT_FP32 );
        /* move and broadcast weight to all lane */
        if ( !NDone && !CDone )
        {
          tpu_gdma_cpy_S2L ( weightAddr, weightGAddr, &WeightShape, NULL, NULL, DT_FP32 );
          tpu_bdc_npu_bcast ( weightAddr, weightAddr, &WeightBCastShape, DT_FP32 );
        }
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
      /* compute normed input & product with grad output */
      dim4 WBcastStride;
      tpu_aligned_stride ( &WBcastStride, 0, &ParamShape, DT_FP32 );
      WBcastStride.w = 0;
      tpu_bdc_fp_sub ( input_normAddr, inputAddr, meanAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      tpu_bdc_fp_mul ( input_normAddr, input_normAddr, invstdAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      tpu_bdc_fp_mul ( prodAddr, input_normAddr, grad_outputAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
      /* pooling prod & grad output in nc-dim */
      dim2 KernelSizeNCdim = { .h = Shape.n, .w = Shape.c };
      dim4 TransStride;
      tpu_aligned_stride ( &TransStride, 0, &TransShape, DT_FP32 );
      dim4 TTransStride = { .n = TransStride.h, .c = TransStride.c, .h = TransStride.n, .w = TransStride.w };
      tpu_bdc_cw_trans ( tmp1Addr, grad_outputAddr, &TransShape, DT_FP32 );
      tpu_bdc_cpy ( tmp2Addr, tmp1Addr, &PoolingShape, NULL, &TTransStride, DT_FP32 );
      tpu_bdc_fp_avg_pool2d ( reducetmpAddr, tmp2Addr, &PoolingShape, &KernelSizeNCdim, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, OneFP );
      if ( !Split || ( !NDone && !CDone ) )
      {
        tpu_bdc_cw_trans ( grad_biasAddr, reducetmpAddr, &WeightShape, DT_FP32 );
      }
      else
      {
        tpu_bdc_cw_trans ( sumtmpAddr, reducetmpAddr, &WeightShape, DT_FP32 );
        tpu_bdc_fp_add ( grad_biasAddr, grad_biasAddr, sumtmpAddr, &WeightShape, NULL, NULL, NULL, DT_FP32 );
      }
      tpu_bdc_cw_trans ( tmp1Addr, prodAddr, &TransShape, DT_FP32 );
      tpu_bdc_cpy ( tmp2Addr, tmp1Addr, &PoolingShape, NULL, &TTransStride, DT_FP32 );
      tpu_bdc_fp_avg_pool2d ( reducetmpAddr, tmp2Addr, &PoolingShape, &KernelSizeNCdim, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, OneFP );
      if ( !Split || ( !NDone && !CDone ) )
      {
        tpu_bdc_cw_trans ( grad_weightAddr, reducetmpAddr, &WeightShape, DT_FP32 );
      }
      else
      {
        tpu_bdc_cw_trans ( sumtmpAddr, reducetmpAddr, &WeightShape, DT_FP32 );
        tpu_bdc_fp_add ( grad_weightAddr, grad_weightAddr, sumtmpAddr, &WeightShape, NULL, NULL, NULL, DT_FP32 );
      }
      /* compute prod = weight * grad output * normed input */
      dim4 NCBcastStride;
      tpu_aligned_stride ( &NCBcastStride, 0, &WeightShape, DT_FP32 );
      NCBcastStride.n = 0;
      NCBcastStride.c = 0;
      tpu_bdc_fp_mul ( grad_outputAddr, grad_outputAddr, weightAddr, &Shape, NULL, NULL, &NCBcastStride, DT_FP32 );
      tpu_bdc_fp_mul ( prodAddr, prodAddr, weightAddr, &Shape, NULL, NULL, &NCBcastStride, DT_FP32 );
      /* pooling prod & grad output in w-dim */
      dim2 KernelSizeWdim = { .h = 1, .w = Shape.w };
      tpu_bdc_fp_avg_pool2d ( bufferAddr, prodAddr, &Shape, &KernelSizeWdim, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, InvW );
      tpu_bdc_fp_mul ( input_normAddr, input_normAddr, bufferAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      tpu_bdc_fp_avg_pool2d ( bufferAddr, grad_outputAddr, &Shape, &KernelSizeWdim, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, InvW );
      tpu_bdc_fp_add ( input_normAddr, input_normAddr, bufferAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      /* compute and move grad input */
      tpu_bdc_fp_sub ( grad_inputAddr, grad_outputAddr, inputAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
      tpu_bdc_fp_mul ( grad_inputAddr, grad_inputAddr, invstdAddr, &Shape, NULL, NULL, &WBcastStride, DT_FP32 );
      if ( dtype != DT_FP32 )
      {
        tpu_bdc_cast ( grad_inputAddr, grad_inputAddr, &Shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
      }
      if ( grad_input_enable ) tpu_gdma_cpy_L2S ( grad_inputGAddr, grad_inputAddr, &Shape, &GlobalStride, NULL, dtype );
      CTodo -= Shape.c;
      CDone += Shape.c;
    }
    NTodo -= Shape.n;
    NDone += Shape.n;
  }
  if ( dtype != DT_FP32 )
  {
    tpu_bdc_cast ( grad_biasAddr, grad_biasAddr, &WeightShape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
    tpu_bdc_cast ( grad_weightAddr, grad_weightAddr, &WeightShape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
  }
  if ( grad_bias_enable ) tpu_gdma_cpy_L2S ( grad_bias_global_addr, grad_biasAddr, &WeightShape, NULL, NULL, dtype );
  if ( grad_weight_enable ) tpu_gdma_cpy_L2S ( grad_weight_global_addr, grad_weightAddr, &WeightShape, NULL, NULL, dtype );
}

void tpu_kernel_api_layernorm_backward ( const void *args )
{
  sg_api_layernorm_backward_t *api = ( sg_api_layernorm_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int inner_num = 1, outer_num = 1;
  for ( int i = 0; i < api->axis; ++i )
  {
    outer_num *= api->shape[i];
  }
  for ( int i = api->axis; i < api->dim; ++i )
  {
    inner_num *= api->shape[i];
  }
  dim4 shape = { .n = 1, .c = outer_num, .h = 1, .w = inner_num };
  tpu_initialize();
  nodechip_layernorm_backward_dim3 (
  api->grad_output_global_addr,
  api->input_global_addr,
  api->weight_global_addr,
  api->mean_global_addr,
  api->rstd_global_addr,
  api->grad_input_global_addr,
  api->grad_weight_global_addr,
  api->grad_bias_global_addr,
  shape,
  ( data_type_t ) api->dtype,
  api->grad_input_global_addr != 0,
  api->grad_weight_global_addr != 0,
  api->grad_bias_global_addr != 0 );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_layernorm_backward );
