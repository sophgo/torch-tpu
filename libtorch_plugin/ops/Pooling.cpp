#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/Pool.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{

std::tuple<Tensor, Tensor> max_pool2d_with_indices_tpu (
const Tensor & self,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  std::tuple<Tensor, Tensor> outputs;
#if 1
  auto outputs_cpu = max_pool2d_with_indices ( self.cpu(), kernel_size, stride, padding, dilation, ceil_mode );
  outputs = std::tuple<Tensor, Tensor> (
            TENSOR_TO_TPU ( std::get<0> ( outputs_cpu ) ),
            TENSOR_TO_TPU ( std::get<1> ( outputs_cpu ) ) );
#else
  TORCH_CHECK ( ceil_mode == false );
  TORCH_CHECK ( dilation[0] == 1 && dilation[1] == 1, "DILATION must be one" );
  float alpha = 1.f;
  float beta = 0.f;
  PoolingDescriptor_t pooling_desc =
  {
    .kh = ( int ) kernel_size[0],
    .kw = ( int ) kernel_size[1],
    .pad_h = ( int ) padding[0],
    .pad_w = ( int ) padding[1],
    .stride_h = ( int ) stride[0],
    .stride_w = ( int ) stride[1],
    .mode = Pooling_MAX
  };
  int output_h = at::native::pooling_output_shape ( self.size ( 2 ), kernel_size[0], padding[0], stride[0], dilation[0], ceil_mode );
  int output_w = at::native::pooling_output_shape ( self.size ( 3 ), kernel_size[1], padding[1], stride[1], dilation[1], ceil_mode );
  auto output = empty ( { self.size ( 0 ), self.size ( 1 ), output_h, output_w }, self.options() );
  bm_status_t status = sgdnn_pooling_forward_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       pooling_desc,
                       &alpha,
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       tpu::TPUGenerateTensorDesc ( output ),
                       ADDR_IN_DEVICE ( output ) );
  TORCH_CHECK ( status == BM_SUCCESS );
  outputs = std::tuple<Tensor, Tensor> ( output, Tensor() );
#endif
  return outputs;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "max_pool2d_with_indices", max_pool2d_with_indices_tpu );
}

Tensor max_pool2d_with_indices_backward_tpu (
const Tensor & grad_output,
const Tensor & self,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor & indices )
{
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( indices );
  Tensor grad_input;
#if 1
  auto grad_input_cpu = max_pool2d_with_indices_backward ( grad_output.cpu(), self.cpu(), kernel_size, stride, padding, dilation, ceil_mode, indices.cpu() );
  grad_input = TENSOR_TO_TPU ( grad_input_cpu );
#else
  TORCH_CHECK ( ceil_mode == false );
  TORCH_CHECK ( dilation[0] == 1 && dilation[1] == 1, "DILATION must be one" );
  float alpha = 1.f;
  float beta = 0.f;
  PoolingDescriptor_t pooling_desc =
  {
    .kh = ( int ) kernel_size[0],
    .kw = ( int ) kernel_size[1],
    .pad_h = ( int ) padding[0],
    .pad_w = ( int ) padding[1],
    .stride_h = ( int ) stride[0],
    .stride_w = ( int ) stride[1],
    .mode = Pooling_MAX
  };
  int output_h = at::native::pooling_output_shape ( self.size ( 2 ), kernel_size[0], padding[0], stride[0], dilation[0], ceil_mode );
  int output_w = at::native::pooling_output_shape ( self.size ( 3 ), kernel_size[1], padding[1], stride[1], dilation[1], ceil_mode );
  auto output = empty ( { self.size ( 0 ), self.size ( 1 ), output_h, output_w }, self.options() );
  bm_status_t status = sgdnn_pooling_forward_cudnn (
                       tpu::TPUGetDeviceHandle(),
                       pooling_desc,
                       &alpha,
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       &beta,
                       tpu::TPUGenerateTensorDesc ( output ),
                       ADDR_IN_DEVICE ( output ) );
  TORCH_CHECK ( status == BM_SUCCESS );
  grad_input = empty ( self.sizes(), self.options() );
  status = sgdnn_pooling_backward_cudnn (
           tpu::TPUGetDeviceHandle(),
           pooling_desc,
           &alpha,
           tpu::TPUGenerateTensorDesc ( output ),
           ADDR_IN_DEVICE ( output ),
           tpu::TPUGenerateTensorDesc ( grad_output ),
           ADDR_IN_DEVICE ( grad_output ),
           tpu::TPUGenerateTensorDesc ( self ),
           ADDR_IN_DEVICE ( self ),
           &beta,
           tpu::TPUGenerateTensorDesc ( grad_input ),
           ADDR_IN_DEVICE ( grad_input ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#endif
  return grad_input;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_tpu );
}
} // namespace at
