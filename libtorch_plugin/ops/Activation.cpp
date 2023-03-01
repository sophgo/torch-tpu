#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <limits.h>

//#define TPU_LIBTORCH_OP_COMPARE
#define TPU_OP_TIMING

namespace at
{

Tensor & relu__tpu ( Tensor & self )
{
  //std::cout << "ReLU" << std::endl;
#if 0
  auto self_cpu = self.to ( torch::Device ( "cpu" ) );
  relu_ ( self_cpu );
  self = self_cpu.to ( tpu::TPUGetCurrentDevice() );
  return self;
#else
  CHECK_TENSOR_IN_DEVICE ( self );
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto self_cpu = self.to ( torch::Device ( "cpu" ) );
  auto & output_exp = relu_ ( self_cpu );
#endif
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
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
  status = sgdnn_activation_forward_cudnn (
           handle,
           activation_desc,
           &alpha,
           self_desc,
           ADDR_IN_DEVICE ( self ),
           &beta,
           self_desc,
           ADDR_IN_DEVICE ( self ) );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RELU, timer.ElapsedUS() );
#endif
#ifdef TPU_LIBTORCH_OP_COMPARE
  std::cout << "Comparing inplace relu:"
            << " self shape = " << self.sizes()
            << " self dtype = " << self.dtype()
            << std::endl;
  auto output_got = self.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( output_got, output_exp );
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
  auto grad_output_cpu = grad_output.to ( torch::Device ( "cpu" ) );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  grad_input = threshold_backward ( grad_output_cpu, input_cpu, threshold );
  grad_input = grad_input.to ( tpu::TPUGetCurrentDevice() );
  return grad_input;
#else
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto grad_output_cpu = grad_output.to ( torch::Device ( "cpu" ) );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto grad_input_exp = threshold_backward ( grad_output_cpu, input_cpu, threshold );
#endif
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
#ifdef TPU_LIBTORCH_OP_COMPARE
  std::cout << "Comparing threshold backward:"
            << " grad_output shape = " << grad_output.sizes()
            << " grad_output dtype = " << grad_output.dtype()
            << " input shape = " << input.sizes()
            << " input dtype = " << input.dtype()
            << " grad_input shape = " << grad_input.sizes()
            << " grad_input dtype = " << grad_input.dtype()
            << " threshold = " << threshold
            << std::endl;
  auto grad_input_got = grad_input.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( grad_input_got, grad_input_exp );
#endif
  return grad_input;
#endif
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "threshold_backward.grad_input", threshold_backward_grad_input_tpu );
}
} // namespace at
