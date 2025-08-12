#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{
Tensor& index_select_out_tpu ( const Tensor & self, int64_t dim, const Tensor & index, Tensor & out)
{
  TIMING_START;
  if ( self.numel() == 0) {return out;}
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( index );
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnIndexSelectAsync(
                stream,
                tpu::TPUGenerateTpudnnTensor(stream, self),
                tpu::TPUGenerateTpudnnTensor(stream, index),
                dim,
                tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(self, index, out);
  return out;
}

Tensor index_select_tpu ( const Tensor & self, int64_t dim, const Tensor & index )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( index );
#if 0
  auto out_cpu = index_select ( self.cpu(), dim, index.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  std::vector<int64_t> sizes_vec;
  for ( int i = 0; i < dim; i++ ) {
    sizes_vec.push_back ( self.size ( i ) );
  }
  for ( int i = 0; i <  index.dim(); i++ ) {
    sizes_vec.push_back ( index.size ( i ) );
  }
  for ( int i = dim + 1; i < self.dim(); i++ ) {
    sizes_vec.push_back ( self.size ( i ) );
  }
  IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
  auto out = torch::empty ( sizes, options );
  index_select_out_tpu(self, dim, index, out);
#endif
  SHOW_TENSOR_OP(self, index, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "index_select.out", index_select_out_tpu );
  m.impl ( "index_select", index_select_tpu );
}

Tensor embedding_dense_backward_tpu ( const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq )
{
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( indices );
#if 0
  auto out_cpu = embedding_dense_backward ( grad_output.cpu(), indices.cpu(), num_weights, padding_idx, scale_grad_by_freq );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
  TensorOptions out_option = TensorOptions ( grad_output.device() ).dtype ( grad_output.dtype() );
  torch::Tensor out = torch::empty ( {num_weights, grad_output.size ( grad_output.dim() - 1 ) }, out_option );
  // indices should not be int64_t
  auto indices_int32 = indices.to ( torch::kInt32 );
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnEmbeddingBackwardAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, grad_output),
    tpu::TPUGenerateTpudnnTensor(stream, indices_int32),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(grad_output, indices, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "embedding_dense_backward", embedding_dense_backward_tpu );
}

Tensor masked_select_tpu(const at::Tensor & self, const at::Tensor & mask) {
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mask );
#if 1
  LOG( WARNING ) << __func__ << " use cpu impl.";
  auto out_cpu = masked_select( self.cpu(), mask.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#endif
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "masked_select", masked_select_tpu );
}

} // namespace at
