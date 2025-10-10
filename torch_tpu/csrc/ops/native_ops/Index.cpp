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
    TIMING_START;
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
      auto idx = indices[0].value();
      auto idx_value = idx.scalar_type() == torch::kInt64 || idx.scalar_type() == torch::kInt32 ? idx : idx.to(torch::kInt32);

      if (!IS_TPU_TENSOR(idx_value))
          idx_value = idx_value.to(self.device());

      auto stream = c10_tpu::getCurrentTPUStream();
      if (self.scalar_type() == torch::kInt64){
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
      }
#endif
    }
#endif
    TIMING_END;
    SHOW_TENSOR_OP(self, out);
    SHOW_TENSOR_OP(indices[0].value());
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor_out",  index_out_tpu);
}

static bool onlyBC_last(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices)
{
  bool ret = true;
  for (size_t i = 0; i < indices.size(); i++)
  {
    if ( indices[i].has_value() && indices[i].value().defined() ) { continue; }
    else { ret = false; break;}
  }
  return ret;
}

Tensor index_tpu(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices)
{
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( self );
  Tensor out;
  if ( indices.size() == 1 )
  {
    auto index_ = indices[0].value().contiguous();
    if ( index_.scalar_type() == at::ScalarType::Int )
    { 
      out = torch::index_select(self, 0, indices[0].value());
    }
    else if ( index_.scalar_type() == at::ScalarType::Bool )
    {
      CPU_IMPL_WARNING();
      // maskselect must malloc mem with mask's content, 
      // have no idea how to avoid break stream, just host hack do.
      c10::List<c10::optional<Tensor>> indices_cpu;
      for (size_t i = 0; i < indices.size(); i++)
      {
          c10::optional<Tensor> indice = c10::nullopt;
          if ( indices[i].has_value() ) {
            if (indices[i].value().defined()) indice = indices[i].value().cpu();
            else                              indice = indices[i].value();
          }
          indices_cpu.push_back(indice);
      }
      auto out_cpu = index( self.cpu(), indices_cpu );
      out = out_cpu.to( self.device());

      // int64_t num_nonzeros = torch::sum( index_.to(torch::kInt32) ).item().toInt();
      // std::vector<int64_t> out_sizes = {num_nonzeros};
      // for (int i = index_.dim(); i < self.dim(); i++) { out_sizes.push_back( self.size(i) ); }
      // out = torch::empty(out_sizes, self_.options());

      // auto stream = c10_tpu::getCurrentTPUStream();
      // auto status = tpudnnMaskedSelectAsync(
      //     stream,
      //     tpu::TPUGenerateTpudnnTensor(stream, self_),
      //     tpu::TPUGenerateTpudnnTensor(stream, index_),
      //     tpu::TPUGenerateTpudnnTensor(stream, out));
      // TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
    else
    {
      TORCH_CHECK( false, "index_tpu should not be called here." )
    }
  }
  else if ( onlyBC_last(self, indices) )
  {
    auto final_index = zeros(indices[0].value().sizes(), indices[0].value().options());
    int64_t C = 1;
    for (int i = (int)indices.size() - 1; i >= 0; i--)
    {
      auto stream = c10_tpu::getCurrentTPUStream();
      auto index_ = indices[i].value().contiguous();
      auto status = tpudnnBinaryAsync(
          stream,
          tpu::TPUGenerateTpudnnTensor(stream, final_index),
          tpu::TPUGenerateTpudnnTensor(stream, index_ ),
          (float)C,
          tpu::TPUGenerateTpudnnTensor(stream, final_index),
          0);
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      C = C * self.size(i);
    }
    std::vector<int64_t> view_sizes = { C };
    for (size_t i = indices.size(); i < (unsigned long)self.dim(); i++) 
    {
      view_sizes.push_back(self.size(i));
    }
    IntArrayRef size_ref( view_sizes.data(), view_sizes.size());
    auto self_view = self.view(size_ref);
    out = torch::index_select(self_view, 0, final_index);
  }
  else
  {
    CPU_IMPL_WARNING();
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (size_t i = 0; i < indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) {
          if (indices[i].value().defined()) indice = indices[i].value().cpu();
          else                              indice = indices[i].value();
        }

        indices_cpu.push_back(indice);
    }
    auto out_cpu = index( self.cpu(), indices_cpu );
    out = out_cpu.to( self.device());
  }
  TIMING_END;
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index.Tensor",  index_tpu);
}

Tensor & index_put_tpu(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
  SHOW_TENSOR_OP(self, value);
  CHECK_TENSOR_IN_DEVICE ( self );
#if 1
  CPU_IMPL_WARNING();
  c10::List<c10::optional<Tensor>> indices_cpu;
  for (int i = 0; i < (int)indices.size(); i++)
  {
      c10::optional<Tensor> indice = c10::nullopt;
      if ( indices[i].has_value() ) {
        if ( indices[i].value().defined() ) indice = indices[i].value().cpu();
        else                                indice = indices[i].value();
      }
      indices_cpu.push_back(indice);
  }
  auto self_cpu = self.cpu();
  auto out_cpu = index_put(self_cpu, indices_cpu, value.cpu(), accumulate);
  self.copy_(out_cpu);
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
    c10::List<c10::optional<Tensor>> indices_cpu;
    for (int i = 0; i < (int)indices.size(); i++)
    {
        c10::optional<Tensor> indice = c10::nullopt;
        if ( indices[i].has_value() ) {
          if (indices[i].value().defined()) indice = indices[i].value().cpu();
          else                              indice = indices[i].value();
        }
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

Tensor & index_put_impl_tpu(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  TIMING_START;
  SHOW_TENSOR_OP(self, value);
  CHECK_TENSOR_IN_DEVICE ( self );
#if 1
  CPU_IMPL_WARNING();
  c10::List<c10::optional<Tensor>> indices_cpu;
  for (size_t i = 0; i < indices.size(); i++)
  {
      c10::optional<Tensor> indice = c10::nullopt;
      if ( indices[i].has_value() ) {
        if ( indices[i].value().defined() ) indice = indices[i].value().cpu();
        else                                indice = indices[i].value();
      }
      indices_cpu.push_back(indice);
  }
  auto self_cpu = self.cpu();
  auto out_cpu = index_put(self_cpu, indices_cpu, value.cpu(), accumulate);
  tpu::TPUCopyHostToDevice(self.data_ptr(), out_cpu.contiguous().data_ptr(), out_cpu.nbytes());
#else
#endif
  TIMING_END;
  return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "_index_put_impl_",  index_put_impl_tpu);
}

Tensor & index_add_tpu( Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, const Scalar &alpha)
{
    TIMING_START;
    TORCH_CHECK( alpha.toFloat() == 1,  "index_add_ 's alpha now only support 1.");
    Tensor index_ = index;
    if(index.dtype()== torch::kInt64) {
      LOG( WARNING ) << "index_add_ index's dtype is int64, please use int32 dtype get better performance.";
      index_ = index.to(torch::kInt32);
    }
    CHECK_TENSOR_IN_DEVICE ( self );
    CHECK_TENSOR_IN_DEVICE( index_ );
    CHECK_TENSOR_IN_DEVICE( source );
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
    TIMING_END;
    return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index_add_",  index_add_tpu);
}

} //namespace at
