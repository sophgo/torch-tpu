#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

std::tuple<Tensor, Tensor> max_pool2d_with_indices_tpu (
const Tensor & self,
IntArrayRef    kernel_size,
IntArrayRef    stride,
IntArrayRef    padding,
IntArrayRef    dilation,
bool           ceil_mode )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  std::cout << "This is max_pool2d_with_indices_tpu" << std::endl;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "max_pool2d_with_indices", max_pool2d_with_indices_tpu );
}
} // namespace at
