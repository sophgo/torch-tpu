#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <TPUModule.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at
{
Tensor index_select_tpu ( const Tensor & self, int64_t dim, const Tensor & index )
{
  CHECK_TENSOR_IN_DEVICE ( self );
#if 1
  auto out_cpu = index_select ( self.cpu(), dim, index.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::INDEX_SELECT, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
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
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "embedding_dense_backward", embedding_dense_backward_tpu );
}
} // namespace at
