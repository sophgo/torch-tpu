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
Tensor & triu_out_tpu( const Tensor & self, int64_t diagonal, Tensor & out)
{
    CHECK_TENSOR_IN_DEVICE ( out );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 0
    CPU_IMPL_WARNING();
    auto out_cpu = triu( self.cpu(), diagonal );
    out = out_cpu.to(out.device());
#else
    TIMING_START;
    bm_status_t status = sgdnnTriangularize(  tpu::TPUGetDeviceHandle(), 
                                    tpu::TPUGenerateSgdnnTensor(self),
                                    1,
                                    diagonal,
                                    tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK ( status == BM_SUCCESS );
  TIMING_END ( tpu::TRIU );
#endif
    SHOW_TENSOR_OP(self, out);
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "triu.out",  triu_out_tpu);
}

}