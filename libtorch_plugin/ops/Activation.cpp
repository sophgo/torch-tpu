#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <limits.h>

#define TPU_OP_TIMING
#define SHOW_OP_INFO

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
  auto self_cpu = self.to ( torch::Device ( "cpu" ) );
  self_cpu = relu_ ( self_cpu );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
  return self;
#else
  float alpha = 1.f;
  float beta = 0.f;
  auto self_desc = tpu::TPUGenerateTensorDesc ( self );
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Relu,
    .NanOpt = Not_Propagate_Nan,
    .coef = std::numeric_limits<double>::max()
  };
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_activation_forward_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       activation_desc,
                       &alpha,
                       self_desc,
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       self_desc,
                       ADDR_IN_DEVICE ( self ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RELU, timer.ElapsedUS() );
#endif
  return self;
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
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
  auto grad_input_cpu = threshold_backward ( TENSOR_TO_CPU ( grad_output ), TENSOR_TO_CPU ( input ), threshold );
  tpu::TPUCopyHostToDevice ( grad_input.data_ptr(), grad_input_cpu.contiguous().data_ptr(), grad_input.nbytes() );
  return grad_input;
#else
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
  float alpha = 1.f;
  float beta = 0.f;
  auto grad_output_desc = tpu::TPUGenerateTensorDesc ( grad_output );
  auto input_desc = tpu::TPUGenerateTensorDesc ( input );
  auto grad_input_desc = tpu::TPUGenerateTensorDesc ( grad_input );
  auto output_desc = grad_output_desc;
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Relu,
    .NanOpt = Not_Propagate_Nan,
    .coef = threshold.toDouble()
  };
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  status = sgdnn_activation_backward_cudnn (
           handle,
           activation_desc,
           &alpha,
           output_desc,
           nullptr,
           grad_output_desc,
           ADDR_IN_DEVICE ( grad_output ),
           input_desc,
           ADDR_IN_DEVICE ( input ),
           &beta,
           grad_input_desc,
           ADDR_IN_DEVICE ( grad_input ) );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RELU_BACKWARD, timer.ElapsedUS() );
#endif
  return grad_input;
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "threshold_backward.grad_input", threshold_backward_grad_input_tpu );
}

Tensor & gelu_out_tpu ( const Tensor    & self,
                        c10::string_view  approximate,
                        Tensor          & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
  float alpha = 1.f;
  float beta = 0.f;
  auto self_desc = tpu::TPUGenerateTensorDesc ( self );
  auto out_desc = tpu::TPUGenerateTensorDesc ( out );
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Gelu,
    .NanOpt = Not_Propagate_Nan,
    .coef = std::numeric_limits<double>::max()
  };
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_activation_forward_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       activation_desc,
                       &alpha,
                       self_desc,
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       out_desc,
                       ADDR_IN_DEVICE ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "gelu.out", gelu_out_tpu );
}

} // namespace at
