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
    if (indices.size() == 1) {
        TIMING_START;
        bm_status_t status = sgdnnIndexSelect(tpu::TPUGetDeviceHandle(),
                                              tpu::TPUGenerateSgdnnTensor(self),
                                              tpu::TPUGenerateSgdnnTensor(indices[0].value()),
                                              0,
                                              tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK ( status == BM_SUCCESS );
        TIMING_END ( tpu::INDEX_SELECT );
    } else {
        IntArrayRef size_ref = {(int64_t)self.strides().size()};
        TensorOptions option = TensorOptions( ).device("cpu").dtype(torch::kInt64);
        Tensor strides = from_blob(self.strides().vec().data(), size_ref, option).clone(); //.to(torch::kFloat32).to(self.device());
        int64_t index_size = -1;
        std::vector<Tensor> indexes;
        for (int i = 0; i < indices.size(); i++)
        {
            // TODO: empty indices or broadcast shape cases
            TORCH_CHECK (indices[i].has_value(), "Does not support empty indices now!" );
            auto size = indices[i].value().numel();
            TORCH_CHECK(size != 0, "Does not support empty indices now!")
            if (index_size == -1) {
                index_size = size;
            } else {
                TORCH_CHECK(size == index_size, "Does not support shape broadcast now!")
            }
            // matmul does not support int; fp32 has problem; use cpu here
            // indexes.push_back(indices[i].value().to(torch::kFloat32).view(-1));
            indexes.push_back(indices[i].value().cpu().to(torch::kInt64).view(-1));
        }
        Tensor self_one_dim = self.view(-1);
        // Tensor index_one_dim = matmul(strides, stack(indexes)).to(torch::kInt32);
        Tensor index_one_dim = matmul(strides, stack(indexes)).to(self.device());
        TIMING_START;
        bm_status_t status = sgdnnIndexSelect(tpu::TPUGetDeviceHandle(),
                                              tpu::TPUGenerateSgdnnTensor(self_one_dim),
                                              tpu::TPUGenerateSgdnnTensor(index_one_dim),
                                              0,
                                              tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK ( status == BM_SUCCESS );
        TIMING_END ( tpu::INDEX_SELECT );
    }
    
    // std::vector< SgdnnTensor_t> Inds;
    // for (int i = 0; i < indices.size(); i++){\
    //     if (indices[i].has_value()) {
    //         Inds.push_back( tpu::TPUGenerateSgdnnTensor(indices[i].value()) );
    //     } else {
    //         SgdnnTensor_t null_tensor = {0};
    //         Inds.push_back(null_tensor);
    //     }
        
    // }
    // TIMING_START;    
    // bm_status_t status = sgdnnMulIndexSelect(tpu::TPUGetDeviceHandle(),
    //                                 tpu::TPUGenerateSgdnnTensor(self),
    //                                 tpu::TPUGenerateSgdnnTensor(out),
    //                                 Inds);
    // TORCH_CHECK ( status == BM_SUCCESS );
    // TIMING_END ( tpu::INDEX_SELECT );
#endif
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor_out",  index_out_tpu);
}
} //namespace at
