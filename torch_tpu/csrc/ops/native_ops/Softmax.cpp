#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"

namespace at
{
Tensor & _log_softmax_out_tpu ( const Tensor & self, int64_t dim, bool half_to_float, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
  TORCH_CHECK ( half_to_float == false );
#if 0
  CPU_IMPL_WARNING();
  TIMING_START;
  auto out_cpu = _log_softmax ( self.cpu(), dim, half_to_float );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  TIMING_END(tpu::CPU_LAYER);
#else
  Tensor self_f = self;
  Tensor out_f = out;
  if (self.dtype() == caffe2::TypeMeta::Make<at::Half>() ||
      self.dtype() == caffe2::TypeMeta::Make<at::BFloat16>()) {
    self_f = self.to(torch::kFloat);
    out_f = out.to(torch::kFloat);
  }

  TIMING_START;
  #ifdef BACKEND_1684X
  auto status = sgdnnLogSoftmax(tpu::TPUGetDeviceHandle(),
                                       tpu::TPUGenerateSgdnnTensor(self_f), dim,
                                       tpu::TPUGenerateSgdnnTensor(out_f));
  TORCH_CHECK(status == BM_SUCCESS);
  tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_f.to(out.dtype()).data_ptr(),
                             out.nbytes());
  #elif defined BACKEND_SG2260
  auto status = sgdnnLogSoftmax(c10_tpu::getCurrentTPUStream(),
                                       tpu::TPUGenerateSgdnnTensor(self_f), dim,
                                       tpu::TPUGenerateSgdnnTensor(out_f));
  TORCH_CHECK(status == tpuRtSuccess);
  tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_f.to(out.dtype()).data_ptr(),
                             out.nbytes());
  #endif
  TIMING_END(tpu::LOGSOFTMAX);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("_log_softmax.out", _log_softmax_out_tpu);
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
  TIMING_START;
  #if defined BACKEND_1684X
  auto status = sgdnnSoftmax (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       dim,
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
  #elif defined BACKEND_SG2260
  auto status = sgdnnSoftmax (
                       c10_tpu::getCurrentTPUStream(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       dim,
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == tpuRtSuccess );
  #endif
  TIMING_END ( tpu::SOFTMAX );
#endif
  SHOW_TENSOR_OP(self, out);
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
  TIMING_START;
  #if defined BACKEND_1684X
  auto status = sgdnnSoftmaxBackward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( grad_output ),
                       tpu::TPUGenerateSgdnnTensor ( output ),
                       dim,
                       tpu::TPUGenerateSgdnnTensor ( grad_input ) );
  TORCH_CHECK ( status == BM_SUCCESS );
  #elif defined BACKEND_SG2260
  auto status = sgdnnSoftmaxBackward (
                       c10_tpu::getCurrentTPUStream(),
                       tpu::TPUGenerateSgdnnTensor ( grad_output ),
                       tpu::TPUGenerateSgdnnTensor ( output ),
                       dim,
                       tpu::TPUGenerateSgdnnTensor ( grad_input ) );
  TORCH_CHECK ( status == tpuRtSuccess );
  #endif
  TIMING_END ( tpu::SOFTMAX_BACKWARD );
#endif
  SHOW_TENSOR_OP(grad_output, output, grad_input);
  return grad_input;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "_softmax_backward_data.out", _softmax_backward_data_out_tpu );
}
} // namespace at
