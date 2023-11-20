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

// copied from torch source and modified
static std::tuple<bool, Tensor> canDispatchToMaskedFill(const Tensor& self, const torch::List<c10::optional<at::Tensor>>& indices,
const Tensor& value){
  if (!(value.numel() == 1 /*&& value.device().is_cpu()*/)){
    return std::make_tuple(false,Tensor());
  }
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const c10::optional<Tensor>& i: indices) {
    if (!i.has_value() || !(*i).defined()){
      num_ind++;
    } else {
      const Tensor &index = *i;
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()){
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(index.size(j) == self.size(srcIdx), "The shape of the mask ", index.sizes(), " at index ", j,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for (const auto i : c10::irange(num_ind, self.ndimension())) {
    (void)i; //Suppress unused variable warning
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}

Tensor & index_put_tpu(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
  CHECK_TENSOR_IN_DEVICE ( self );
#if 0
  LOG( WARNING ) << "index_put_ use cpu impl";
  c10::List<c10::optional<Tensor>> indices_cpu;
  for (int i = 0; i < indices.size(); i++)
  {
      c10::optional<Tensor> indice = c10::nullopt;
      if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
      indices_cpu.push_back(indice);
  }
  auto self_cpu = self.cpu();
  auto out_cpu = index_put(self_cpu, indices_cpu, value.cpu(), accumulate);
  tpu::TPUCopyHostToDevice(self.data_ptr(), out_cpu.contiguous().data_ptr(), out_cpu.nbytes());
#else
  // copied from torch source and modified
  bool return_flag = false;
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      // this function strictly read the data in the ORIGINAL self.addr
      // do not use masked_fill_ here, our impl will make self.addr change
      auto out = self.masked_fill(std::get<1>(masked_fill_dispatch), value.item());
      tpu::TPUCopyDeviceToDevice(self.data_ptr(), out.contiguous().data_ptr(), out.nbytes());
      return_flag = true;
    }
  }

  if (!return_flag) {
  #if 1
    // have no idea how to impl in tpu. use cpu first.
    LOG( WARNING ) << "index_put_ use cpu impl";
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (int i = 0; i < indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
        indices_cpu.push_back(indice);
    }
    auto self_cpu = self.cpu();
    auto out_cpu = index_put(self_cpu, indices_cpu, value.cpu(), accumulate);
    tpu::TPUCopyHostToDevice(self.data_ptr(), out_cpu.contiguous().data_ptr(), out_cpu.nbytes());
  #else
    TORCH_CHECK(false, "Not implemented");
  #endif
 }
#endif
  return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index_put_",  index_put_tpu);
}

} //namespace at
