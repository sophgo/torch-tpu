#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & cat_out_tpu ( const ITensorListRef & tensors, int64_t dim, Tensor & out )
{
  std::vector<TensorDescriptor_t> inputDescs;
  std::vector<const void *> inputs;
  for ( auto tensor : tensors )
  {
    CHECK_TENSOR_IN_DEVICE ( tensor );
    inputDescs.push_back ( tpu::TPUGenerateTensorDesc ( tensor ) );
    inputs.push_back ( ADDR_IN_DEVICE ( tensor ) );
  }
  auto status = sgdnn_concat (
                tpu::TPUGetDeviceHandle(),
                inputDescs.data(),
                inputs.data(),
                inputs.size(),
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                dim );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cat.out", cat_out_tpu );
}
} // namespace at
