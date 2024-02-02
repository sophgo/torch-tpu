#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at 
{
Tensor & arange_start_out_tpu( const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) 
{
    CHECK_TENSOR_IN_DEVICE ( out );
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = arange(start,end,step);
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    if ((start.toInt() >= 0 && end.toInt() >= 0)){
        int empty_length = (end.toInt()-start.toInt() - 1) / step.toInt() + 1;
        out = empty({empty_length},out.options());
        TIMING_START;
        #if defined BACKEND_1684X
        auto status = sgdnnArange ( tpu::TPUGetDeviceHandle(),
                                            start.toInt(),
                                            end.toInt(),
                                            step.toInt(),
                                            tpu::TPUGenerateSgdnnTensor ( out ));
        TORCH_CHECK ( status == BM_SUCCESS );
        #elif defined BACKEND_SG2260
        auto status = sgdnnArange ( c10_tpu::getCurrentTPUStream(),
                                            start.toInt(),
                                            end.toInt(),
                                            step.toInt(),
                                            tpu::TPUGenerateSgdnnTensor ( out ));
        TORCH_CHECK ( status == tpuRtSuccess );
        #endif
        TIMING_END(tpu::ARANGE)
    }else{
        CPU_IMPL_WARNING();
        TIMING_START;
        auto out_cpu = arange(start,end,step);
        out = out_cpu.to(out.device()).to(out.dtype());
        TIMING_END(tpu::CPU_LAYER);
    }
#endif
    SHOW_TENSOR_OP(out);
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "arange.start_out",  arange_start_out_tpu);
}
}
