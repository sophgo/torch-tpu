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
Tensor index_select_tpu ( const Tensor & self, int64_t dim, const Tensor & index )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( index );
#if 0
  auto out_cpu = index_select ( self.cpu(), dim, index.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
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
  bm_status_t status = sgdnnIndexSelect (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( index ),
                       dim,
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::INDEX_SELECT, timer.ElapsedUS() );
#endif
#endif
  SHOW_TENSOR_OP(self, index, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "index_select", index_select_tpu );
}

Tensor embedding_dense_backward_tpu ( const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( indices );
#if 0
  auto out_cpu = embedding_dense_backward ( grad_output.cpu(), indices.cpu(), num_weights, padding_idx, scale_grad_by_freq );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
  TensorOptions out_option = TensorOptions ( grad_output.device() ).dtype ( grad_output.dtype() );
  torch::Tensor out = torch::empty ( {num_weights, grad_output.size ( grad_output.dim() - 1 ) }, out_option );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnEmbeddingBackward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( grad_output ),
                       tpu::TPUGenerateSgdnnTensor ( indices ),
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::EMBEDDING_BACKWARD, timer.ElapsedUS() );
#endif
#endif
  SHOW_TENSOR_OP(grad_output, indices, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "embedding_dense_backward", embedding_dense_backward_tpu );
}
} // namespace at
