#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & mean_out_tpu ( const Tensor & self, OptionalIntArrayRef dim_opt, bool keepdim, c10::optional<ScalarType> dtype_opt, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
  auto out_cpu = mean ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mean.out", mean_out_tpu );
}

Tensor & sum_IntList_out_tpu ( const Tensor & self, OptionalIntArrayRef dim_opt, bool keepdim, c10::optional<ScalarType> dtype_opt, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
  auto out_cpu = sum ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "sum.IntList_out", sum_IntList_out_tpu );
}
} // namespace at
