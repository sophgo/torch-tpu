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
Tensor & cat_out_tpu ( const ITensorListRef & tensors, int64_t dim, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( out );
  std::vector<TensorDescriptor_t> inputDescs;
  std::vector<const void *> inputs;
  std::vector<Tensor> contiguous_tensors;
  for ( auto tensor : tensors )
  {
    CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
    contiguous_tensors.push_back ( tensor.contiguous() );
    inputDescs.push_back ( tpu::TPUGenerateTensorDesc ( contiguous_tensors.back() ) );
    inputs.push_back ( ADDR_IN_DEVICE ( contiguous_tensors.back() ) );
  }
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto status = sgdnn_concat (
                tpu::TPUGetDeviceHandle(),
                inputDescs.data(),
                inputs.data(),
                inputs.size(),
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                dim );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::CONCAT, timer.ElapsedUS() );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cat.out", cat_out_tpu );
}
} // namespace at