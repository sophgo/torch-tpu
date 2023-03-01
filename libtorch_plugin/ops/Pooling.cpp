#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

std::tuple<Tensor, Tensor> max_pool2d_with_indices_tpu (
const Tensor & input,
IntArrayRef    kernel_size,
IntArrayRef    stride,
IntArrayRef    padding,
IntArrayRef    dilation,
bool           ceil_mode )
{
  //std::cout << "Maxpooling" << std::endl;
  CHECK_TENSOR_IN_DEVICE ( input );
  std::tuple<Tensor, Tensor> outputs;
#if 1
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto outputs_cpu = max_pool2d_with_indices (
                     input_cpu,
                     kernel_size,
                     stride,
                     padding,
                     dilation,
                     ceil_mode );
  outputs = std::tuple<Tensor, Tensor> (
            std::get<0> ( outputs_cpu ).to ( tpu::TPUGetCurrentDevice() ),
            std::get<1> ( outputs_cpu ).to ( tpu::TPUGetCurrentDevice() ) );
#else
  LOG ( FATAL ) << "TPU max_pool2d_with_indices is not implemented";
#endif
  return outputs;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "max_pool2d_with_indices", max_pool2d_with_indices_tpu );
}

Tensor max_pool2d_with_indices_backward_tpu (
const Tensor & grad_output,
const Tensor & input,
IntArrayRef    kernel_size,
IntArrayRef    stride,
IntArrayRef    padding,
IntArrayRef    dilation,
bool           ceil_mode,
const Tensor & indices )
{
  //std::cout << "Maxpooling Backward" << std::endl;
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( input );
  CHECK_TENSOR_IN_DEVICE ( indices );
  auto grad_output_cpu = grad_output.to ( torch::Device ( "cpu" ) );
  auto input_cpu = input.to ( torch::Device ( "cpu" ) );
  auto indices_cpu = indices.to ( torch::Device ( "cpu" ) );
  auto grad_input_cpu = max_pool2d_with_indices_backward (
                        grad_output_cpu,
                        input_cpu,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        ceil_mode,
                        indices_cpu );
  auto grad_input = grad_input_cpu.to ( tpu::TPUGetCurrentDevice() );
  return grad_input;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_tpu );
}
} // namespace at
