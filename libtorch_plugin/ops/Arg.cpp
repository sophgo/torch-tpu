#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace at 
{
Tensor & argmax_out_tpu( const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out)
{
    CHECK_TENSOR_IN_DEVICE ( out );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 1
    LOG( WARNING ) << "argmax use cpu impl";
    auto out_cpu = argmax( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    //TODO
#endif
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "argmax.out",  argmax_out_tpu);
}
}