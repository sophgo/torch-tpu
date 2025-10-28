#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIndexing.h>

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

Tensor & index_put_impl_tpu(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( self );
  
  // If indices are empty, return self
  if (indices.empty()) {
    return self;
  }
  
  if (indices.size() == 1) {
    auto indices_tensor = indices[0].value();
    
    // Check for unsupported cases in single-dim indexing
    bool need_cpu_fallback = false;
    std::string fallback_reason;
    
    // Check if self is not contiguous (e.g., a view with non-standard stride)
    // tpudnnIndexPutImpl may not handle non-contiguous tensors correctly
    if (!self.is_contiguous()) {
      need_cpu_fallback = true;
      fallback_reason = "non-contiguous tensor (view) not supported for single-dim indexing";
    }
    
    // Check if index is multi-dimensional (not supported by TPU)
    if (indices_tensor.dim() > 1) {
      need_cpu_fallback = true;
      fallback_reason = "multi-dimensional index tensor not supported";
    }
    
    // Use CPU fallback for unsupported cases
    if (need_cpu_fallback) {
      CPU_IMPL_WARNING();
      TIMING_END;
      
      // Perform index_put on CPU and copy back
      // Using .copy_() ensures correct handling of views and non-contiguous tensors
      auto self_cpu = self.cpu();
      c10::List<c10::optional<Tensor>> indices_cpu;
      indices_cpu.push_back(indices_tensor.cpu());
      auto value_cpu = value.cpu();
      self_cpu.index_put_(indices_cpu, value_cpu, accumulate);
      
      // Copy back to TPU using PyTorch's copy_ which handles views correctly
      self.copy_(self_cpu);
      return self;
    }

    Tensor index_;
    Tensor value_ = value;

    // Handle bool mask: convert to integer indices
    if (indices_tensor.dtype() == torch::kBool) {
      index_ = torch::nonzero(indices_tensor.cpu()).squeeze(1).to(torch::kInt32).to(indices_tensor.device());
      int64_t num_true = index_.size(0);

      // Broadcast value if needed
      if (value.dim() == 0) {
        std::vector<int64_t> new_shape;
        new_shape.push_back(num_true);
        for (int64_t i = 1; i < self.dim(); i++) {
          new_shape.push_back(self.size(i));
        }
        value_ = value.unsqueeze(0).expand(new_shape).contiguous();
      } else if (value.size(0) != num_true) {
        auto value_shape = value.sizes().vec();
        value_shape[0] = num_true;
        value_ = value.expand(value_shape).contiguous();
      }
    }
    // Support int64 with warning, auto-convert to int32
    else if (indices_tensor.dtype() == torch::kInt64) {
      LOG(WARNING) << "index_put: converting int64 indices to int32. "
                   << "Please use int32 indices for better performance.";
      index_ = indices_tensor.to(torch::kInt32);
    }
    else {
      TORCH_CHECK(indices_tensor.dtype() == torch::kInt32, 
                  "index_put only supports int32, int64, or bool indices, but got ", 
                  indices_tensor.dtype());
      index_ = indices_tensor;
    }
    
    // Handle scalar value: broadcast to match index size
    if (value.dim() == 0) {
      std::vector<int64_t> new_shape;
      new_shape.push_back(index_.size(0));
      for (int64_t i = 1; i < self.dim(); i++) {
        new_shape.push_back(self.size(i));
      }
      value_ = value.unsqueeze(0).expand(new_shape).contiguous();
    }

    CHECK_TENSOR_IN_DEVICE(index_);
    CHECK_TENSOR_IN_DEVICE(value_);

    // Handle empty indices - no operation needed
    if (index_.numel() == 0) {
      return self;
    }

    // Use TPU DNN interface for hardware acceleration
    auto stream = c10_tpu::getCurrentTPUStream();
    tpudnnStatus_t status = tpudnnIndexPutImpl(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, index_),
        tpu::TPUGenerateTpudnnTensor(stream, value_),
        0,  // axis
        accumulate);
    TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
  } else {
    // Multi-dimensional indexing
    
    // Step 1: Check for unsupported cases and collect non-null indices
    std::vector<Tensor> index_tensors;
    std::vector<int64_t> dims;
    bool need_cpu_fallback = false;
    std::string fallback_reason;
    
    for (size_t i = 0; i < indices.size(); i++) {
      if (!indices[i].has_value() || !indices[i].value().defined()) {
        // Found a None index
        need_cpu_fallback = true;
        fallback_reason = "None indices detected";
        break;
      } else {
        auto idx = indices[i].value();
        
        // Check for multi-dimensional index tensors
        if (idx.dim() > 1) {
          need_cpu_fallback = true;
          fallback_reason = "multi-dimensional index tensors not supported";
          break;
        }
        
        // Handle scalar indices (0-dim tensors) - expand to 1-dim
        if (idx.dim() == 0) {
          idx = idx.unsqueeze(0);
        }
        
        // Support int64 with warning, auto-convert to int32
        if (idx.dtype() == torch::kInt64) {
          LOG(WARNING) << "index_put: converting int64 indices to int32 at dimension " << i 
                       << ". Please use int32 indices for better performance.";
          idx = idx.to(torch::kInt32);
        }
        // Support bool indices by converting to int32
        else if (idx.dtype() == torch::kBool) {
          idx = idx.nonzero().squeeze(-1).to(torch::kInt32);
        }
        
        TORCH_CHECK(idx.dtype() == torch::kInt32, 
                    "index_put only supports int32, int64, or bool indices, but got ", idx.dtype(), 
                    " at dimension ", i);

        index_tensors.push_back(idx);
        dims.push_back(i);
      }
    }
    
    // Use CPU fallback for unsupported cases
    if (need_cpu_fallback) {
      CPU_IMPL_WARNING();
      TIMING_END;
      
      // Perform index_put on CPU and copy back
      // Using .copy_() ensures correct handling of views and non-contiguous tensors
      auto self_cpu = self.cpu();
      c10::List<c10::optional<Tensor>> indices_cpu;
      for (size_t i = 0; i < indices.size(); i++) {
        c10::optional<Tensor> indice = c10::nullopt;
        if (indices[i].has_value() && indices[i].value().defined()) {
          indice = indices[i].value().cpu();
        }
        indices_cpu.push_back(indice);
      }
      auto value_cpu = value.cpu();
      self_cpu.index_put_(indices_cpu, value_cpu, accumulate);
      
      // Copy back to TPU using PyTorch's copy_ which handles views correctly
      self.copy_(self_cpu);
      return self;
    }
    
    // No unsupported cases - proceed with TPU acceleration
    TORCH_CHECK(!dims.empty(), "index_put requires at least one non-None index");
    
    // Check all indices and value are on TPU device
    for (const auto& idx : index_tensors) {
      CHECK_TENSOR_IN_DEVICE(idx);
    }
    CHECK_TENSOR_IN_DEVICE(value);

    // Step 2: Determine reshape strategy and compute appropriate strides
    auto shape = self.sizes();
    int64_t last_indexed_dim = dims.back();
    Tensor flat_self;
    std::vector<int64_t> strides;
    
    if (last_indexed_dim == static_cast<int64_t>(self.dim()) - 1) {
      // All dimensions are indexed - complete flattening
      flat_self = self.view(-1);
      
      // Compute element-based strides for complete indexing
      // For complete indexing, we compute strides for all dimensions
      strides.resize(shape.size());
      int64_t stride = 1;
      for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
      }
    } else {
      // Partial indexing - reshape to (indexed_part, unindexed_part)
      int64_t indexed_size = 1;
      int64_t remaining_size = 1;
      
      for (int64_t i = 0; i <= last_indexed_dim; i++) {
        indexed_size *= shape[i];
      }
      for (int64_t i = last_indexed_dim + 1; i < static_cast<int64_t>(shape.size()); i++) {
        remaining_size *= shape[i];
      }
      
      flat_self = self.view({indexed_size, remaining_size});
      
      // Compute row-based strides for partial indexing
      // For partial indexing, we want flat_index to point to rows in the reshaped tensor
      // The reshaped tensor has shape [indexed_size, remaining_size]
      // We need stride[i] such that flat_index = sum(idx[i] * stride[i])
      // where flat_index points to the correct row in the reshaped tensor
      
      // For shape [2, 3, 4, 5] with dims [0, 1]:
      // Reshaped to [6, 20] where indexed_size = 2*3 = 6
      // self[0, 1, :, :] should map to row 0*3 + 1 = 1
      // self[1, 2, :, :] should map to row 1*3 + 2 = 5
      // So stride[0] = 3, stride[1] = 1
      
      // Compute strides in the correct order
      // For dims = [0, 1] and shape = [2, 3, 4, 5]:
      // We want strides[0] = 3 (for dim 0), strides[1] = 1 (for dim 1)
      // So flat_index = idx0 * 3 + idx1 * 1
      
      strides.resize(dims.size());
      int64_t stride = 1;
      for (int64_t i = last_indexed_dim; i >= 0; i--) {
        // Find which position in dims this dimension corresponds to
        for (size_t j = 0; j < dims.size(); j++) {
          if (dims[j] == i) {
            strides[j] = stride;
            break;
          }
        }
        stride *= shape[i];
      }
    }

    // Step 3: Compute flat index using the computed strides
    Tensor flat_index = index_tensors[0] * static_cast<int>(strides[0]);
    for (size_t i = 1; i < index_tensors.size(); i++) {
      flat_index = flat_index + index_tensors[i] * static_cast<int>(strides[i]);
    }
    
    // Handle empty indices - no operation needed
    if (flat_index.numel() == 0) {
      return self;
    }

    // Handle scalar value broadcasting for multi-dimensional indexing
    Tensor value_ = value;
    if (value.dim() == 0) {
      // For scalar values, we need to broadcast to match the number of indexed elements
      int64_t num_elements = flat_index.size(0);
      std::vector<int64_t> value_shape;
      
      if (last_indexed_dim == static_cast<int64_t>(self.dim()) - 1) {
        // All dimensions indexed - value should be scalar or match num_elements
        value_shape = {num_elements};
      } else {
        // Partial indexing - value should match (num_elements, remaining_dims...)
        value_shape = {num_elements};
        for (int64_t i = last_indexed_dim + 1; i < static_cast<int64_t>(self.dim()); i++) {
          value_shape.push_back(self.size(i));
        }
      }
      
      value_ = value.unsqueeze(0).expand(value_shape).contiguous();
    } else {
      // For non-scalar values, verify the shape matches expectations
      int64_t num_elements = flat_index.size(0);
      std::vector<int64_t> expected_shape;
      
      if (last_indexed_dim == static_cast<int64_t>(self.dim()) - 1) {
        // All dimensions indexed - value should match num_elements
        expected_shape = {num_elements};
      } else {
        // Partial indexing - value should match (num_elements, remaining_dims...)
        expected_shape = {num_elements};
        for (int64_t i = last_indexed_dim + 1; i < static_cast<int64_t>(self.dim()); i++) {
          expected_shape.push_back(self.size(i));
        }
      }
      
      // Check if value shape matches expected shape
      bool shape_matches = (value.dim() == static_cast<int64_t>(expected_shape.size()));
      if (shape_matches) {
        for (int64_t i = 0; i < value.dim(); i++) {
          if (value.size(i) != expected_shape[i]) {
            shape_matches = false;
            break;
          }
        }
      }
      
      if (!shape_matches) {
        TORCH_CHECK(false, "Value tensor shape ", value.sizes(), 
                   " does not match expected shape ", expected_shape,
                   " for multi-dimensional indexing");
      }
    }

    // Use TPU DNN interface for hardware acceleration
    auto stream = c10_tpu::getCurrentTPUStream();
    tpudnnStatus_t status = tpudnnIndexPutImpl(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, flat_self),
        tpu::TPUGenerateTpudnnTensor(stream, flat_index),
        tpu::TPUGenerateTpudnnTensor(stream, value_),
        0,  // axis
        accumulate);
    TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
    // Note: flat_self is a view of self, changes propagate automatically
  }

  SHOW_TENSOR_OP(self, value);
  TIMING_END;
  return self;
}


TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "_index_put_impl_",  index_put_impl_tpu);
}

Tensor & index_put_tpu(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
  // Delegate to index_put_impl_tpu with unsafe=false
  return index_put_impl_tpu(self, indices, value, accumulate, false);
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "index_put_",  index_put_tpu);
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
