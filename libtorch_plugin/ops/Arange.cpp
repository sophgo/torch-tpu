#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"

namespace at 
{
Tensor & arange_start_out_tpu( const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) 
{
    CHECK_TENSOR_IN_DEVICE ( out );
#if 1
    LOG( WARNING ) << "arrange use cpu impl";
    auto out_cpu = arange(start,end,step);
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    int empty_length=std::ceil((end.toInt()-start.toInt())/step.toInt());
    out = empty({empty_length},out.options().dtype(torch::kInt32));
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnArange ( tpu::TPUGetDeviceHandle(),
                                        start.toInt(),
                                        end.toInt(),
                                        step.toInt(),
                                        tpu::TPUGenerateSgdnnTensor ( out ));
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::ARANGE, timer.ElapsedUS() );
#endif
#endif
    return out;
    }

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "arange.start_out",  arange_start_out_tpu);
}
}
