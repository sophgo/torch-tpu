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
Tensor & index_out_tpu( const Tensor & self, const c10::List<c10::optional<Tensor>> & indices, Tensor & out)
{
    CHECK_TENSOR_IN_DEVICE ( out );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 1
    LOG( WARNING ) << "index use cpu impl";
    auto out_cpu = index( self.cpu(), indices );
    out = out_cpu.to(out.device());
#else
    //TODO
#endif
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor_out",  index_out_tpu);
}
} //namespace at
