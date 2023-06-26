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
  float alpha = 1.f;
  float beta = 0.f;
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Relu,
    .NanOpt = Not_Propagate_Nan,
    .coef = std::numeric_limits<double>::max()
  };
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_activation_forward (
                       tpu::TPUGetDeviceHandle(),
                       activation_desc,
                       &alpha,
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ) );
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
  float alpha = 1.f;
  float beta = 0.f;
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Relu,
    .NanOpt = Not_Propagate_Nan,
    .coef = threshold.toDouble()
  };
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_activation_backward (
                       tpu::TPUGetDeviceHandle(),
                       activation_desc,
                       &alpha,
                       tpu::TPUGenerateTensorDesc ( grad_output ),
                       nullptr,
                       tpu::TPUGenerateTensorDesc ( grad_output ),
                       ADDR_IN_DEVICE ( grad_output ),
                       tpu::TPUGenerateTensorDesc ( input ),
                       ADDR_IN_DEVICE ( input ),
                       &beta,
                       tpu::TPUGenerateTensorDesc ( grad_input ),
                       ADDR_IN_DEVICE ( grad_input ) );
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
  float alpha = 1.f;
  float beta = 0.f;
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Gelu,
    .NanOpt = Not_Propagate_Nan,
    .coef = std::numeric_limits<double>::max()
  };
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_activation_forward (
                       tpu::TPUGetDeviceHandle(),
                       activation_desc,
                       &alpha,
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ) );
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
  bm_status_t status = sgdnn_gelu_backward (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( grad_output ),
                       ADDR_IN_DEVICE ( grad_output ),
                       tpu::TPUGenerateTensorDesc ( grad_input ),
                       ADDR_IN_DEVICE ( grad_input ) );
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
