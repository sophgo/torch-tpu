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
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_local_scalar_dense", _local_scalar_dense_tpu );
}

Tensor scalar_tensor_tpu (const Scalar& s,
                          c10::optional<ScalarType> dtype,
                          c10::optional<Layout> layout,
                          c10::optional<Device> device,
                          c10::optional<bool> pin_memory)
{
  printf("using scalar_tensor_tpu\n");
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  return empty({}, options).fill_(s);
}
TORCH_LIBRARY_IMPL (aten, TPU, m)
{
  m.impl ("scalar_tensor", scalar_tensor_tpu);
}


} // namespace at
