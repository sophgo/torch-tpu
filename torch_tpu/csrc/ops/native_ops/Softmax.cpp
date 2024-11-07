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

  TIMING_START;
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnLogSoftmaxAsync(
                stream,
                tpu::TPUGenerateTpudnnTensor ( stream, self ),
                dim,
                tpu::TPUGenerateTpudnnTensor ( stream, out ) );
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS );
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

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnSoftmaxAsync(
                      stream,
                      tpu::TPUGenerateTpudnnTensor ( stream,self ),
                      dim,
                      tpu::TPUGenerateTpudnnTensor ( stream,out) );
  TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS  );
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

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnSoftmaxBackwardAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, grad_output),
      tpu::TPUGenerateTpudnnTensor(stream, output),
      dim,
      tpu::TPUGenerateTpudnnTensor(stream, grad_input));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
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
