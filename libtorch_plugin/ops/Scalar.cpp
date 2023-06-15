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
Scalar _local_scalar_dense_tpu ( const Tensor & self )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  return _local_scalar_dense ( TENSOR_TO_CPU ( self ) );
}
//TORCH_LIBRARY_IMPL ( aten, TPU, m )
//{
//  m.impl ( "_local_scalar_dense", _local_scalar_dense_tpu );
//}
} // namespace at
