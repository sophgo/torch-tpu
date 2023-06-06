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
Tensor & _log_softmax_out_tpu ( const Tensor & self, int64_t dim, bool half_to_float, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
  TORCH_CHECK ( half_to_float == false );
  auto out_cpu = _log_softmax ( self.cpu(), dim, half_to_float );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_log_softmax.out", _log_softmax_out_tpu );
}

#if 0
Tensor & _log_softmax_backward_data_out_tpu ( const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType input_dtype, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( out );
  auto out_cpu = _log_softmax_backward_data ( grad_output.cpu(), output.cpu(), dim, input_dtype );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_log_softmax_backward_data.out", _log_softmax_backward_data_out_tpu );
}
#endif

Tensor & _softmax_out_tpu ( const Tensor & self, int64_t dim, bool half_to_float, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
  TORCH_CHECK ( half_to_float == false );
#if 0
  auto out_cpu = _softmax ( self.cpu(), dim, half_to_float );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  float alpha = 1.f;
  float beta = 0.f;
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_softmax_forward_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       dim < 0 ? dim + self.dim() : dim,
                       &alpha,
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::SOFTMAX, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_softmax.out", _softmax_out_tpu );
}

Tensor & _softmax_backward_data_out_tpu ( const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType input_dtype, Tensor & grad_input )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( output );
  CHECK_TENSOR_IN_DEVICE ( grad_input );
  // TORCH_CHECK ( input_dtype == kFloat );
#if 0
  auto grad_input_cpu = _softmax_backward_data ( grad_output.cpu(), output.cpu(), dim, input_dtype );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_softmax_backward_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       dim < 0 ? dim + output.dim() : dim,
                       tpu::TPUGenerateTensorDesc ( output ),
                       ADDR_IN_DEVICE ( output ),
                       tpu::TPUGenerateTensorDesc ( grad_output ),
                       ADDR_IN_DEVICE ( grad_output ),
                       tpu::TPUGenerateTensorDesc ( grad_input ),
                       ADDR_IN_DEVICE ( grad_input ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::SOFTMAX_BACKWARD, timer.ElapsedUS() );
#endif
#endif
  return grad_input;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_softmax_backward_data.out", _softmax_backward_data_out_tpu );
}
} // namespace at
