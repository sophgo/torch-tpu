#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <TPUModule.h>
#include <sgdnn_api.h>

namespace at
{
Tensor index_select_tpu ( const Tensor & Input,
                          long           Dim,
                          const Tensor & Index )
{
  CHECK_TENSOR_IN_DEVICE ( Input );
  auto InputCPU = TENSOR_TO_CPU ( Input );
  auto IndexCPU = TENSOR_TO_CPU ( Index );
  auto OutputCPU = index_select ( InputCPU, Dim, IndexCPU );
  return TENSOR_TO_TPU ( OutputCPU );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "index_select", index_select_tpu );
}
} // namespace at
