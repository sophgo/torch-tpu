#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

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
  bm_status_t status = sgdnn_index_select_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( index ),
                       ADDR_IN_DEVICE ( index ),
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ),
                       dim );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::INDEX_SELECT, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "index_select", index_select_tpu );
}



Tensor embedding_dense_backward_tpu ( const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  auto out_cpu = embedding_dense_backward ( grad_output.cpu(), indices.cpu(), num_weights, padding_idx, scale_grad_by_freq );
  auto out = TENSOR_TO_TPU ( out_cpu );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "embedding_dense_backward", embedding_dense_backward_tpu );
}
} // namespace at
