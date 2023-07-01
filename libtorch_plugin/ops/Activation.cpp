#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <limits.h>

#include "common/config.h"

namespace at
{

Tensor & relu__tpu ( Tensor & self )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "ReLU " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE ( self );
#if 0
  auto self_cpu = self.cpu();
  self_cpu = relu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnReLU ( tpu::TPUGetDeviceHandle(),
                                   tpu::TPUGenerateSgdnnTensor ( self ),
                                   tpu::TPUGenerateSgdnnTensor ( self ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RELU, timer.ElapsedUS() );
#endif
  return self;
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "relu_", relu__tpu );
}

Tensor relu_tpu ( const Tensor & self )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "ReLU " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE ( self );
#if 0
  auto self_cpu = self.cpu();
  self_cpu = relu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
  auto out = empty ( self.sizes(), self.options() );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnReLU ( tpu::TPUGetDeviceHandle(),
                                   tpu::TPUGenerateSgdnnTensor ( self ),
                                   tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RELU, timer.ElapsedUS() );
#endif
  return out;
#endif
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "relu", relu_tpu );
}

Tensor & threshold_backward_grad_input_tpu (
const Tensor & grad_output,
const Tensor & input,
const Scalar & threshold,
Tensor       & grad_input )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( grad_input );
#if 0
  auto grad_input_cpu = threshold_backward ( grad_output.cpu(), input.cpu(), threshold );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnReLUBackward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( grad_output ),
                       tpu::TPUGenerateSgdnnTensor ( input ),
                       tpu::TPUGenerateSgdnnTensor ( grad_input ) );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RELU_BACKWARD, timer.ElapsedUS() );
#endif
#endif
  return grad_input;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "threshold_backward.grad_input", threshold_backward_grad_input_tpu );
}

Tensor & gelu_out_tpu ( const Tensor & self, c10::string_view approximate, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = gelu ( self.cpu(), approximate );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnGELU ( tpu::TPUGetDeviceHandle(),
                                   tpu::TPUGenerateSgdnnTensor ( self ),
                                   tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::GELU, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "gelu.out", gelu_out_tpu );
}

Tensor & gelu_backward_grad_input_tpu ( const Tensor & grad_output, const Tensor & self, c10::string_view approximate, Tensor & grad_input )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( grad_input );
#if 0
  auto grad_input_cpu = gelu_backward ( grad_output.cpu(), self.cpu(), approximate );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnGELUBackward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( grad_output ),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( grad_input ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::GELU_BACKWARD, timer.ElapsedUS() );
#endif
#endif
  return grad_input;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "gelu_backward.grad_input", gelu_backward_grad_input_tpu );
}
} // namespace at
