#include "sg_api_struct.h"
#include "tpu_kernel.h"

static inline void nodechip_softmax_backward_dim3 (
global_addr_t output_global_addr,
global_addr_t grad_output_global_addr,
global_addr_t grad_input_global_addr,
dim4 shape,
data_type_t dtype )
{
  const int N = shape.n;
  const int C = shape.c;
  const int H = shape.h;
  const int W = shape.w;
  const dim4 TotalShape = { .n = N, .c = C, .h = H, .w = W };
  dim4 GlobalStride;
  tpu_continuous_stride ( &GlobalStride, &TotalShape );
  const padding_t ZeroPadding = { .top = 0, .bottom = 0, .left = 0, .right = 0 };
  const dim2 OneStride = { .h = 1, .w = 1 };
  const dim2 OneDilation = { .h = 1, .w = 1 };
  const scalar_t OneFP = { .f32 = 1.f };
  const int DataSize = tpu_data_type_size ( dtype );
  /*
  *  grad_output : [ NMax, CMax, HMax,     W ]
  *  output      : [ NMax, CMax, HMax,     W ]
  *  sumprod     : [ NMax, CMax, HMax,     1 ]
  */
  int NMax = N, CMax = C, HMax = H;
  local_addr_t outputCast, grad_outputCast;
  local_addr_t outputAddr, grad_outputAddr, sumprodAddr;
  local_addr_t grad_inputAddr, prodAddr;
  while ( true )
  {
    grad_outputAddr = 0;
    if ( dtype != DT_FP32 )
    {
      outputCast = 0;
      int InSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, dtype );
      grad_outputCast = outputCast + InSize;
      grad_outputAddr = grad_outputCast + InSize;
    }
    int grad_outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, DT_FP32 );
    outputAddr = grad_outputAddr + grad_outputSize;
    int outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, DT_FP32 );
    sumprodAddr = outputAddr + outputSize;
    int sumprodSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, 1, DT_FP32 );
    grad_inputAddr = grad_outputAddr;
    prodAddr = grad_outputAddr;
    if ( ( int ) sumprodAddr + sumprodSize <= LOCAL_MEM_SIZE ) { break; }
    else
    {
      if ( NMax > 1 )
      {
        NMax /= 2;
        continue;
      }
      else if ( CMax > NPU_NUM )
      {
        CMax -= NPU_NUM;
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
  int NTodo = N, NDone = 0;
  while ( NTodo > 0 )
  {
    Shape.n = MIN ( NTodo, NMax );
    int CTodo = C, CDone = 0;
    while ( CTodo > 0 )
    {
      Shape.c = MIN ( CTodo, CMax );
      int HTodo = H, HDone = 0;
      while ( HTodo > 0 )
      {
        Shape.h = MIN ( HTodo, HMax );
        int TotalDone = ( NDone * GlobalStride.n + CDone * GlobalStride.c + HDone * GlobalStride.h ) * DataSize;
        global_addr_t grad_inputGAddr = grad_input_global_addr + TotalDone;
        global_addr_t grad_outputGAddr = grad_output_global_addr + TotalDone;
        global_addr_t outputGAddr = output_global_addr + TotalDone;
        if ( dtype != DT_FP32 )
        {
          tpu_gdma_cpy_S2L ( grad_outputCast, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
          tpu_gdma_cpy_S2L ( outputCast, outputGAddr, &Shape, NULL, &GlobalStride, dtype );
          tpu_bdc_cast ( outputAddr, outputCast, &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
          tpu_bdc_cast ( grad_outputAddr, grad_outputCast, &Shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN );
        }
        else
        {
          tpu_gdma_cpy_S2L ( grad_outputAddr, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
          tpu_gdma_cpy_S2L ( outputAddr, outputGAddr, &Shape, NULL, &GlobalStride, dtype );
        }
        /* compute product of grad_output & output */
        tpu_bdc_fp_mul ( prodAddr, outputAddr, grad_outputAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
        /* compute sum of product */
        dim2 KernelSize = { .h = 1, .w = Shape.w };
        tpu_bdc_fp_avg_pool2d ( sumprodAddr, prodAddr, &Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, DT_FP32, OneFP );
        /* compute output mul sum of product */
        dim4 sumprodStride;
        dim4 sumprodShape = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
        tpu_aligned_stride ( &sumprodStride, 0, &sumprodShape, DT_FP32 );
        sumprodStride.w = 0;
        tpu_bdc_fp_mul ( outputAddr, outputAddr, sumprodAddr, &Shape, NULL, NULL, &sumprodStride, DT_FP32 );
        /* compute grad_input */
        tpu_bdc_fp_sub ( grad_inputAddr, prodAddr, outputAddr, &Shape, NULL, NULL, NULL, DT_FP32 );
        if ( dtype != DT_FP32 )
        {
          tpu_bdc_cast ( grad_inputAddr, grad_inputAddr, &Shape, NULL, NULL, dtype, DT_FP32, RM_HALF_TO_EVEN );
        }
        tpu_gdma_cpy_L2S ( grad_inputGAddr, grad_inputAddr, &Shape, &GlobalStride, NULL, dtype );
        HTodo -= Shape.h;
        HDone += Shape.h;
      }
      CTodo -= Shape.c;
      CDone += Shape.c;
    }
    NTodo -= Shape.n;
    NDone += Shape.n;
  }
}


void tpu_kernel_api_softmax_backward ( const void *args )
{
  sg_api_softmax_backward_t *api = ( sg_api_softmax_backward_t * ) args;
  tpu_initialize();
  TPUKERNEL_ASSERT ( api->axis   == api->dim - 1 );
  int len = 1;
  for ( int i = 0; i < api->dim - 1; ++i )
  {
    len *= api->shape[i];
  }
  dim4 shape = {1, len, 1, api->shape[api->dim - 1]};
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  nodechip_softmax_backward_dim3 (
  api->output_global_addr,
  api->grad_output_global_addr,
  api->grad_input_global_addr,
  shape,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_softmax_backward );
