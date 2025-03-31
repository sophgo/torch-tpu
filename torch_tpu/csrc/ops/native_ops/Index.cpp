#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{
Tensor & index_out_tpu( const Tensor & self, const c10::List<c10::optional<Tensor>> & indices, Tensor & out)
{
    CHECK_TENSOR_IN_DEVICE ( out );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 0
    CPU_IMPL_WARNING();
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (size_t i = 0; i < indices.size(); i++)
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
      auto idx = indices[0].value();
      auto idx_value = idx.scalar_type() == torch::kInt64 || idx.scalar_type() == torch::kInt32 ? idx : idx.to(torch::kInt32);

      if (!IS_TPU_TENSOR(idx_value))
          idx_value = idx_value.to(self.device());

      auto stream = c10_tpu::getCurrentTPUStream();
      if (tpu::TPUConvertDtype<SgdnnDataType_t>(self.dtype()) == SGDNN_DTYPE_INT64){
        auto self_ = self.to(torch::kInt32);
        auto out_  = out.to(torch::kInt32);
        auto status = tpudnnIndexSelectAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, self_),
            tpu::TPUGenerateTpudnnTensor(stream, idx_value),
            0,
            tpu::TPUGenerateTpudnnTensor(stream, out_));
        TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
        out = out_.to(torch::kInt64);
      }
      else{
        auto status = tpudnnIndexSelectAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, self),
            tpu::TPUGenerateTpudnnTensor(stream, idx_value),
            0,
            tpu::TPUGenerateTpudnnTensor(stream, out));
        TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
      }

      TIMING_END ( tpu::INDEX_SELECT );
    } else {
    CPU_IMPL_WARNING();
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (size_t i = 0; i < indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
        indices_cpu.push_back(indice);
    }
    auto out_cpu = index( self.cpu(), indices_cpu );
    out = out_cpu.to(out.device());
#if 0
      int64_t index_size = -1;
      std::vector<bool> broadcast;
      std::vector<int> broadcast_shape;
      std::vector<int> no_broadcast_shape;

      std::vector<Tensor> indexes;
      for (size_t i = 0; i < indices.size(); ++i) {
        // check empty index (dim broadcast)
        auto size = indices[i].value().numel();
        if (!indices[i].has_value() or size == 0) {
          broadcast.push_back(true);
          broadcast_shape.push_back(self.size(i));
        } else {
          broadcast.push_back(false);
          no_broadcast_shape.push_back(self.size(i));
          // check index shape broadcast
          if (index_size == -1) {
            index_size = size;
          } else {
            TORCH_CHECK(size == index_size, "Does not support shape broadcast now!")
          }
          // matmul does not support int; fp32 has problem; use cpu here
          // indexes.push_back(indices[i].value().to(torch::kFloat32).view(-1));
          indexes.push_back(indices[i].value().cpu().to(torch::kInt64).view(-1));
        }
      }
      for (int i = indices.size(); i < self.dim(); ++i) {
        broadcast.push_back(true);
        broadcast_shape.push_back(self.size(i));
      }
      // only support broadcast at beginning or end; in the middle not support
      int change = 0;
      int start = 0;
      int end = self.dim();
      for (size_t i = 0; i < broadcast.size() - 1; ++i) {
        if (broadcast[i] != broadcast[i+1]) {
          ++change;
          if (broadcast[i])
            start = i + 1;
          else
            end = i + 1;
        }
      }
      if (change <= 1) {
        std::vector<int64_t> stride_data;
        for (auto i = start; i < end; ++i) {
          stride_data.push_back(std::accumulate(self.sizes().begin() + i + 1, self.sizes().begin() + end, 1, std::multiplies<int64_t>()));
        }
        IntArrayRef size_ref = {(int64_t)stride_data.size()};
        TensorOptions option = TensorOptions( ).device("cpu").dtype(torch::kInt64);
        Tensor strides = from_blob(stride_data.data(), size_ref, option).clone(); //.to(torch::kFloat32).to(self.device());

        auto dim0 = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<int>());
        auto dim1 = std::accumulate(no_broadcast_shape.begin(), no_broadcast_shape.end(), 1, std::multiplies<int>());
        Tensor self_two_dim = broadcast[0] ? self.view({dim0, dim1}) : self.view({dim1, dim0});
        Tensor index_two_dim = matmul(strides, stack(indexes)).to(torch::kInt32).to(self.device());
        Tensor out_two_dim = broadcast[0] ? out.view({dim0, -1}) : out.view({-1, dim0});
        TIMING_START;

        auto status = sgdnnIndexSelect(tpu::TPUGetDeviceResource(),
                                              tpu::TPUGenerateSgdnnTensor(self_two_dim),
                                              tpu::TPUGenerateSgdnnTensor(index_two_dim),
                                              (int)broadcast[0],
                                              tpu::TPUGenerateSgdnnTensor(out_two_dim));

        TORCH_CHECK ( status == SG_SUCCESS );
                TIMING_END ( tpu::INDEX_SELECT );
      } else {
        // no support; use cpu impl
        CPU_IMPL_WARNING();
        TIMING_START;
        c10::List<c10::optional<Tensor>> indices_cpu;
        for (int i = 0; i < (int)indices.size(); i++)
        {
            c10::optional<Tensor> indice = c10::nullopt;
            if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
            indices_cpu.push_back(indice);
        }
        auto out_cpu = index( self.cpu(), indices_cpu );
        out = out_cpu.to(out.device());
        TIMING_END(tpu::CPU_LAYER);
      }
#endif
    }
#endif
    SHOW_TENSOR_OP(self, out);
    SHOW_TENSOR_OP(indices[0].value());
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor_out",  index_out_tpu);
}

Tensor index_tpu(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices)
{
  CHECK_TENSOR_IN_DEVICE ( self );
  Tensor out;
  if ( indices.size() == 1 )
  {
    auto idx = indices[0].value();
    std::vector<int64_t> sizes_vec;
    sizes_vec.push_back ( idx.size( 0 ) );
    for ( int i = 1; i <  self.dim(); i++ ) {
      sizes_vec.push_back ( self.size ( i ) );
    }

    IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
    TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
    out = torch::empty( sizes, options);
    out = index_out_tpu(self, indices, out);
  }
  else
  {
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (size_t i = 0; i < indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
        indices_cpu.push_back(indice);
    }
    auto out_cpu = index( self.cpu(), indices_cpu );
    out = out_cpu.to( self.device());
  }

  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor",  index_tpu);
}

// copied from torch source and modified
static std::tuple<bool, Tensor> canDispatchToMaskedFill(const Tensor& self, const torch::List<c10::optional<at::Tensor>>& indices,
const Tensor& value){
  SHOW_TENSOR_OP(self);
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
  SHOW_TENSOR_OP(self, value);
  CHECK_TENSOR_IN_DEVICE ( self );
#if 0
  CPU_IMPL_WARNING();
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
    CPU_IMPL_WARNING();
    TIMING_START;
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (int i = 0; i < (int)indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
        indices_cpu.push_back(indice);
    }
    auto self_cpu = self.cpu();
    auto out_cpu = index_put(self_cpu, indices_cpu, value.cpu(), accumulate);
    tpu::TPUCopyHostToDevice(self.data_ptr(), out_cpu.contiguous().data_ptr(), out_cpu.nbytes());
    TIMING_END(tpu::CPU_LAYER);
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

Tensor & index_put_impl_tpu(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  SHOW_TENSOR_OP(self, value);
  CHECK_TENSOR_IN_DEVICE ( self );
#if 1
  CPU_IMPL_WARNING();
  c10::List<c10::optional<Tensor>> indices_cpu;
  for (size_t i = 0; i < indices.size(); i++)
  {
      c10::optional<Tensor> indice = c10::nullopt;
      if ( indices[i].has_value() ) { indice = indices[i].value().cpu(); }
      indices_cpu.push_back(indice);
  }
  auto self_cpu = self.cpu();
  auto out_cpu = index_put(self_cpu, indices_cpu, value.cpu(), accumulate);
  tpu::TPUCopyHostToDevice(self.data_ptr(), out_cpu.contiguous().data_ptr(), out_cpu.nbytes());
#else
#endif
  return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "_index_put_impl_",  index_put_impl_tpu);
}

Tensor & index_add_tpu( Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, const Scalar &alpha)
{
    TORCH_CHECK( alpha.toFloat() == 1,  "index_add_ 's alpha now only support 1.");
    Tensor index_ = index;
    if(index.dtype()== torch::kInt64) {
      LOG( WARNING ) << "index_add_ index's dtype is int64, please use int32 dtype get better performance.";
      index_ = index.to(torch::kInt32);
    }
    CHECK_TENSOR_IN_DEVICE ( self );
    CHECK_TENSOR_IN_DEVICE( index_ );
    CHECK_TENSOR_IN_DEVICE( source );
    TIMING_START;
#if defined BACKEND_SG2260
    auto stream = c10_tpu::getCurrentTPUStream();

    tpudnnStatus_t status = tpudnnIndexAdd(
                            stream,
                            tpu::TPUGenerateTpudnnTensor(stream, self),
                            tpu::TPUGenerateTpudnnTensor(stream, index_),
                            tpu::TPUGenerateTpudnnTensor(stream, source),
                            dim);
    TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
      // }
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
    SHOW_TENSOR_OP(self, index, source);
    TIMING_END ( tpu::INDEX_ADD_ );
    return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index_add_",  index_add_tpu);
}

} //namespace at
