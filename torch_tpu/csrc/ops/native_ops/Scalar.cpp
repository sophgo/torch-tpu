#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{
Scalar _local_scalar_dense_tpu ( const Tensor & self )
{
  TIMING_START;
  SHOW_TENSOR_OP(self);
  CHECK_TENSOR_IN_DEVICE ( self );
  Scalar out;
#if 1
  CPU_IMPL_WARNING();
  out = _local_scalar_dense ( TENSOR_TO_CPU ( self ) );
#else
#endif
  TIMING_END;
  return out;
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
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  if ( options.dtype() == torch::kInt64 ) {
    LOG( WARNING ) << "scalar_tensor_tpu detect int64 dtype," << "auto cast to int32 dtype.";
    options = options.dtype( ScalarType::Int );
  }
  return empty({}, options).fill_(s);
}
TORCH_LIBRARY_IMPL (aten, TPU, m)
{
  m.impl ("scalar_tensor", scalar_tensor_tpu);
}


} // namespace at
