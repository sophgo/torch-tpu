#include "sg_api_struct.h"
#include "tpu_kernel.h"

static inline void nodechip_batchnorm2d_backward (
global_addr_t grad_output_global_addr,
global_addr_t input_global_addr,
global_addr_t weight_global_addr,
global_addr_t saved_mean_global_addr,
global_addr_t saved_invstd_global_addr,
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
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const int CPerNPU = DIV_UP ( C, NPU_NUM );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.f };
  const scalar_t InvNHW = { .f32 = 1.0 / ( ( double ) N * H * W ) };
  const int DataSize = tpu_data_type_size ( dtype );
  /*
  *  grad_output    : [ NMax, CMax, HMax,     W ]
  *  input & prod   : [ NMax, CMax, HMax,     W ]
  *  input_norm     : [ NMax, CMax, HMax,     W ]
  *  pooling_tmp1   : [ NMax, CMax, HMax,     1 ]
  *  pooling_tmp2   : [ NMax, CMax,    1,     1 ]
  *  grad_bias_tmp  : [ 1,    CMax,    1,  NMax ]
  *  grad_weight_tmp: [ 1,    CMax,    1,  NMax ]
  *  grad_bias      : [ 1,    CMax,    1,     1 ]
  *  grad_weight    : [ 1,    CMax,    1,     1 ]
  *  weight         : [ 1,    CMax,    1,     1 ]
  *  mean           : [ 1,    CMax,    1,     1 ]
  *  invstd         : [ 1,    CMax,    1,     1 ]
  */
  int HMax = H, NMax = N, CMax = CPerNPU * NPU_NUM;
  local_addr_t grad_inputAddr, grad_outputAddr;
  local_addr_t inputAddr, input_normAddr, prodAddr;
  local_addr_t pooling_tmp1, pooling_tmp2;
  local_addr_t grad_weight_tmp, grad_bias_tmp;
  local_addr_t grad_biasAddr, grad_weightAddr;
  local_addr_t meanAddr, invstdAddr, weightAddr;
  bool Split_N_or_H = false;
  while ( true )
  {
    Split_N_or_H = NMax != N || HMax != H ;
    grad_outputAddr = 0;
    int grad_outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, dtype );
    inputAddr = grad_outputAddr + grad_outputSize;
    int inputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, dtype );
    if ( Split_N_or_H )
    {
      input_normAddr = inputAddr;
    }
    else
    {
      input_normAddr = inputAddr + inputSize;
    }
    int input_normSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, dtype );
    pooling_tmp1 = input_normAddr + input_normSize;
    int poolingSize_dim3 = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, 1, dtype );
    pooling_tmp2 = pooling_tmp1 + poolingSize_dim3;
    int poolingSize_dim2 = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
    grad_bias_tmp = pooling_tmp2 + poolingSize_dim2;
    int tmpSize1 = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, NMax, dtype );
    grad_weight_tmp = grad_bias_tmp + tmpSize1;
    int tmpSize2 = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, NMax, dtype );
    grad_biasAddr = grad_weight_tmp + tmpSize2;
    int cSize = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1,  1, dtype );
    grad_weightAddr = grad_biasAddr + cSize;
    weightAddr = grad_weightAddr + cSize;
    meanAddr = weightAddr + cSize;
    invstdAddr = meanAddr + cSize;
    grad_inputAddr = grad_outputAddr;
    prodAddr = inputAddr;
    if ( ( int ) invstdAddr + cSize <= LOCAL_MEM_SIZE ) { break; }
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
    global_addr_t meanGAddr = saved_mean_global_addr + CDone * DataSize;
    global_addr_t invstdGAddr = saved_invstd_global_addr + CDone * DataSize;
    global_addr_t weightGAddr = weight_global_addr + CDone * DataSize;
    tpu_gdma_cpy_S2L ( meanAddr, meanGAddr, &CShape, NULL, NULL, dtype );
    tpu_gdma_cpy_S2L ( invstdAddr, invstdGAddr, &CShape, NULL, NULL, dtype );
    tpu_gdma_cpy_S2L ( weightAddr, weightGAddr, &CShape, NULL, NULL, dtype );
    /*
     * compute grad weight & bias :
     * grad_bias   = pooling ( grad_output, dim=(n,h,w) )
     * grad_weight = pooling ( grad_output * input_norm, dim=(n,h,w) )
     */
    int NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      dim4 poolingShapeDim2 = { .n = Shape.n, .c = Shape.c, .h = 1, .w = 1 };
      dim4 transedShapeDim2 = { .n = 1, .c = Shape.c, .h = 1, .w = Shape.n };
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        int TotalDone = ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
        global_addr_t grad_outputGAddr = grad_output_global_addr + TotalDone;
        global_addr_t inputGAddr = input_global_addr + TotalDone;
        tpu_gdma_cpy_S2L ( grad_outputAddr, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
        tpu_gdma_cpy_S2L ( inputAddr, inputGAddr, &Shape, NULL, &GlobalStride, dtype );
        /* compute input_norm */
        tpu_bdc_fp_sub ( input_normAddr, inputAddr, meanAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        tpu_bdc_fp_mul ( input_normAddr, input_normAddr, invstdAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        /* compute grad_output * input_norm */
        tpu_bdc_fp_mul ( prodAddr, input_normAddr, grad_outputAddr, &Shape, NULL, NULL, NULL, dtype );
        dim2 KernelSizeDim3 = { .h = 1, .w = Shape.w };
        dim2 KernelSizeDim2 = { .h = Shape.h, .w = 1 };
        dim4 poolingShapeDim3 = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
        dim4 poolingStrideDim2;
        tpu_aligned_stride ( &poolingStrideDim2, 0, &poolingShapeDim2, dtype );
        dim4 transedStride = { .n = poolingStrideDim2.w, .c = poolingStrideDim2.c, .h = poolingStrideDim2.h, .w = poolingStrideDim2.n };
        /* compute grad_bias */
        tpu_bdc_fp_avg_pool2d ( pooling_tmp1, grad_outputAddr, &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        tpu_bdc_fp_avg_pool2d ( pooling_tmp2, pooling_tmp1, &poolingShapeDim3, &KernelSizeDim2, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        if ( HDone == 0 )
        {
          tpu_bdc_cpy ( grad_bias_tmp, pooling_tmp2, &transedShapeDim2, NULL, &transedStride, dtype );
        }
        else
        {
          tpu_bdc_fp_add ( grad_bias_tmp, grad_bias_tmp, pooling_tmp2, &transedShapeDim2, NULL, NULL, &transedStride, dtype );
        }
        /* compute grad_weight */
        tpu_bdc_fp_avg_pool2d ( pooling_tmp1, prodAddr, &Shape, &KernelSizeDim3, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        tpu_bdc_fp_avg_pool2d ( pooling_tmp2, pooling_tmp1, &poolingShapeDim3, &KernelSizeDim2, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
        if ( HDone == 0 )
        {
          tpu_bdc_cpy ( grad_weight_tmp, pooling_tmp2, &transedShapeDim2, NULL, &transedStride, dtype );
        }
        else
        {
          tpu_bdc_fp_add ( grad_weight_tmp, grad_weight_tmp, pooling_tmp2, &transedShapeDim2, NULL, NULL, &transedStride, dtype );
        }
        HTodo -= Shape.h;
        HDone += Shape.h;
      }
      dim2 KernelSizeDim1 = { .h = transedShapeDim2.h, .w = transedShapeDim2.w };
      tpu_bdc_fp_avg_pool2d ( NDone == 0 ? grad_biasAddr : pooling_tmp2, grad_bias_tmp, &transedShapeDim2, &KernelSizeDim1, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( NDone > 0 )
      {
        tpu_bdc_fp_add ( grad_biasAddr, grad_biasAddr, pooling_tmp2, &CShape, NULL, NULL, NULL, dtype );
      }
      tpu_bdc_fp_avg_pool2d ( NDone == 0 ? grad_weightAddr : pooling_tmp2, grad_weight_tmp, &transedShapeDim2, &KernelSizeDim1, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
      if ( NDone > 0 )
      {
        tpu_bdc_fp_add ( grad_weightAddr, grad_weightAddr, pooling_tmp2, &CShape, NULL, NULL, NULL, dtype );
      }
      NTodo -= Shape.n;
      NDone += Shape.n;
    }
    if ( grad_bias_enable )
    {
      global_addr_t grad_biasGAddr = grad_bias_global_addr + CDone * DataSize;
      tpu_gdma_cpy_L2S ( grad_biasGAddr, grad_biasAddr, &CShape, NULL, NULL, dtype );
    }
    if ( grad_weight_enable )
    {
      global_addr_t grad_weightGAddr = grad_weight_global_addr + CDone * DataSize;
      tpu_gdma_cpy_L2S ( grad_weightGAddr, grad_weightAddr, &CShape, NULL, NULL, dtype );
    }
    /* compute weight * invstd & weight * invstd * invnhw */
    tpu_bdc_fp_mul ( weightAddr, weightAddr, invstdAddr, &CShape, NULL, NULL, NULL, dtype );
    /*
     * compute grad input :
     * part1 = weight * invstd * grad_output
     * part2 = weight * invstd * invnhw
     * grad_input = part1 - part2 * (input_norm * grad_weight + grad_bias)
     */
    NTodo = N, NDone = 0;
    while ( NTodo > 0 )
    {
      Shape.n = MIN ( NTodo, NMax );
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        int TotalDone = ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
        if ( Split_N_or_H )
        {
          global_addr_t grad_outputGAddr = grad_output_global_addr + TotalDone;
          global_addr_t inputGAddr = input_global_addr + TotalDone;
          tpu_gdma_cpy_S2L ( grad_outputAddr, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
          tpu_gdma_cpy_S2L ( inputAddr, inputGAddr, &Shape, NULL, &GlobalStride, dtype );
          tpu_bdc_fp_sub ( input_normAddr, inputAddr, meanAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
          tpu_bdc_fp_mul ( input_normAddr, input_normAddr, invstdAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        }
        /* compute weight * invstd * grad_output */
        tpu_bdc_fp_mul ( grad_outputAddr, grad_outputAddr, weightAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        /* compute input_norm * grad_weight + grad_bias */
        tpu_bdc_fp_mul ( input_normAddr, input_normAddr, grad_weightAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        tpu_bdc_fp_add ( input_normAddr, input_normAddr, grad_biasAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        tpu_bdc_fp_mul ( input_normAddr, input_normAddr, weightAddr, &Shape, NULL, NULL, &CBcastStride, dtype );
        tpu_bdc_fp_mul_C ( input_normAddr, input_normAddr, InvNHW, &Shape, NULL, NULL, dtype );
        /* compute grad input */
        tpu_bdc_fp_sub ( grad_inputAddr, grad_outputAddr, input_normAddr, &Shape, NULL, NULL, NULL, dtype );
        if ( grad_input_enable )
        {
          global_addr_t grad_inputGAddr = grad_input_global_addr + TotalDone * DataSize;
          tpu_gdma_cpy_L2S ( grad_inputGAddr, grad_inputAddr, &Shape, NULL, NULL, dtype );
        }
        else { TPUKERNEL_ASSERT ( false ); }
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

void tpu_kernel_api_batchnorm2d_backward ( const void *args )
{
  sg_api_batchnorm2d_backward_t *api = ( sg_api_batchnorm2d_backward_t * ) args;
  dim4 shape = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};
  // TODO:fp16 to fp32 local type convert
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  tpu_initialize();
  nodechip_batchnorm2d_backward (
  api->grad_output_global_addr,
  api->input_global_addr,
  api->weight_global_addr,
  api->saved_mean_global_addr,
  api->saved_invstd_global_addr,
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
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_batchnorm2d_backward );
