#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

static inline void nodechip_softmax_backward_dim3(
    global_addr_t output_global_addr,
    global_addr_t grad_output_global_addr,
    global_addr_t grad_input_global_addr,
    dim4 shape,
    data_type_t dtype)
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
    local_addr_t outputAddr, grad_outputAddr, sumprodAddr;
    local_addr_t grad_inputAddr, prodAddr;
    while(true)
    {
        grad_outputAddr = 0;
        int grad_outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, dtype );
        outputAddr = grad_outputAddr + grad_outputSize;
        int outputSize = NMax * DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( HMax, W, dtype );
        sumprodAddr = outputAddr + outputSize;
        int sumprodSize = DIV_UP ( CMax, NPU_NUM ) * tpu_aligned_feature_size ( 1, 1, dtype );
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
                tpu_gdma_cpy_S2L ( grad_outputAddr, grad_outputGAddr, &Shape, NULL, &GlobalStride, dtype );
                tpu_gdma_cpy_S2L ( outputAddr, outputGAddr, &Shape, NULL, &GlobalStride, dtype );
                /* compute product of grad_output & output */
                tpu_bdc_fp_mul ( prodAddr, outputAddr, grad_outputAddr, &Shape, NULL, NULL, NULL, dtype );
                /* compute sum of product */
                dim2 KernelSize = { .h = 1, .w = Shape.w };
                tpu_bdc_fp_avg_pool2d ( sumprodAddr, prodAddr, &Shape, &KernelSize, &ZeroPadding, &OneStride, &OneDilation, dtype, OneFP );
                /* compute output mul sum of product */
                dim4 sumprodStride;
                dim4 sumprodShape = { .n = Shape.n, .c = Shape.c, .h = Shape.h, .w = 1 };
                tpu_aligned_stride ( &sumprodStride, 0, &sumprodShape, dtype );
                sumprodStride.w = 0;
                tpu_bdc_fp_mul ( outputAddr, outputAddr, sumprodAddr, &Shape, NULL, NULL, &sumprodStride, dtype );
                /* compute grad_input */
                tpu_bdc_fp_sub ( grad_inputAddr, prodAddr, outputAddr, &Shape, NULL, NULL, NULL, dtype );
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


void tpu_kernel_api_softmax_backward(const void *args)
{
    sg_api_softmax_backward_t *api = (sg_api_softmax_backward_t *)args;
    tpu_initialize();
    
    TPUKERNEL_ASSERT ( api->dtype == SG_DTYPE_FP32);
    TPUKERNEL_ASSERT ( api->dim   == 3);
    dim4 shape = {api->input_n, api->input_c, api->input_h, api->input_w};
    
    nodechip_softmax_backward_dim3(
        api->output_global_addr,
        api->grad_output_global_addr,
        api->grad_input_global_addr,
        shape,
        tpu_type_convert(api->dtype));
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_softmax_backward);