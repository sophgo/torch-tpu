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
#if 0
    LOG( WARNING ) << "index use cpu impl";
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (int i = 0; i < indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
        indices_cpu.push_back(indice);
    }
    auto out_cpu = index( self.cpu(), indices_cpu );
    out = out_cpu.to(out.device());
#else
    std::vector< SgdnnTensor_t> Inds;
    for (int i = 0; i < indices.size(); i++){
        Inds.push_back( tpu::TPUGenerateSgdnnTensor(indices[i].value()) );
    }
    TIMING_START;    
    bm_status_t status = sgdnnMulIndexSelect(tpu::TPUGetDeviceHandle(),
                                    tpu::TPUGenerateSgdnnTensor(self),
                                    tpu::TPUGenerateSgdnnTensor(out),
                                    Inds);
    TORCH_CHECK ( status == BM_SUCCESS );
    TIMING_END ( tpu::INDEX_SELECT );
#endif
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor_out",  index_out_tpu);
}
} //namespace at
