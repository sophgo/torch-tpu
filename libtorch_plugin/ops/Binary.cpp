#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

namespace at
{
Tensor & add_out_tpu ( const Tensor & input1,
                       const Tensor & input2,
                       const Scalar & alpha,
                       Tensor       & out )
{
#if 0
  auto out_cpu = torch::mul ( input1.to ( torch::Device ( "cpu" ) ), input2.to ( torch::Device ( "cpu" ) ) );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto handle = tpu::TPUGetDeviceHandle();
  if ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::PrivateUse1 )
  {
    CHECK_TENSOR_IN_DEVICE ( input1 );
    CHECK_TENSOR_IN_DEVICE ( input2 );
    CHECK_TENSOR_IN_DEVICE ( out );
    auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
    auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
    auto output_desc = tpu::TPUGenerateTensorDesc ( out );
    if ( alpha.toDouble() == 1.0 )
    {
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input1_desc,
                           ADDR_IN_DEVICE ( input1 ),
                           input2_desc,
                           ADDR_IN_DEVICE ( input2 ),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
    else
    {
      auto input2_mul_alpha = input2 * alpha;
      input2_desc = tpu::TPUGenerateTensorDesc ( input2_mul_alpha );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input1_desc,
                           ADDR_IN_DEVICE ( input1 ),
                           input2_desc,
                           ADDR_IN_DEVICE ( input2_mul_alpha ),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
  }
  else if ( ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::CPU ) ||
            ( input1.device().type() == DeviceType::CPU && input2.device().type() == DeviceType::PrivateUse1 ) )
  {
    TORCH_CHECK ( alpha.toDouble() == 1.0 );
    if ( input2.device().type() == DeviceType::CPU )
    {
      TORCH_CHECK ( input2.dim() == 0, "Input2 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input1 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input2.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input2.to ( torch::kFloat );
      }
      else
      {
        Scalar = input2;
      }
      auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input1_desc,
                           ADDR_IN_DEVICE ( input1 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
    else
    {
      TORCH_CHECK ( input1.dim() == 0, "Input1 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input2 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input1.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input1.to ( torch::kFloat );
      }
      else
      {
        Scalar = input1;
      }
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input2_desc,
                           ADDR_IN_DEVICE ( input2 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_ADD );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "add.out", add_out_tpu );
}

Tensor & sub_out_tpu ( const Tensor & input1,
                       const Tensor & input2,
                       const Scalar & alpha,
                       Tensor       & out )
{
  TORCH_CHECK ( alpha.toDouble() == 1.0 );
#if 0
  auto out_cpu = torch::mul ( input1.to ( torch::Device ( "cpu" ) ), input2.to ( torch::Device ( "cpu" ) ) );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto handle = tpu::TPUGetDeviceHandle();
  if ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::PrivateUse1 )
  {
    CHECK_TENSOR_IN_DEVICE ( input1 );
    CHECK_TENSOR_IN_DEVICE ( input2 );
    CHECK_TENSOR_IN_DEVICE ( out );
    auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
    auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
    auto output_desc = tpu::TPUGenerateTensorDesc ( out );
    bm_status_t status = sgdnn_binary_cudnn (
                         handle,
                         input1_desc,
                         ADDR_IN_DEVICE ( input1 ),
                         input2_desc,
                         ADDR_IN_DEVICE ( input2 ),
                         output_desc,
                         ADDR_IN_DEVICE ( out ),
                         OP_BINARY_SUB );
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else if ( ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::CPU ) ||
            ( input1.device().type() == DeviceType::CPU && input2.device().type() == DeviceType::PrivateUse1 ) )
  {
    if ( input2.device().type() == DeviceType::CPU )
    {
      TORCH_CHECK ( input2.dim() == 0, "Input2 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input1 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input2.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input2.to ( torch::kFloat );
      }
      else
      {
        Scalar = input2;
      }
      auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input1_desc,
                           ADDR_IN_DEVICE ( input1 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_SUB );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
    else
    {
      TORCH_CHECK ( input1.dim() == 0, "Input1 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input2 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input1.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input1.to ( torch::kFloat );
      }
      else
      {
        Scalar = input1;
      }
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input2_desc,
                           ADDR_IN_DEVICE ( input2 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_SUB );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "sub.out", sub_out_tpu );
}

Tensor & mul_out_tpu ( const Tensor & input1,
                       const Tensor & input2,
                       Tensor       & out )
{
#if 0
  auto out_cpu = torch::mul ( input1.to ( torch::Device ( "cpu" ) ), input2.to ( torch::Device ( "cpu" ) ) );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto handle = tpu::TPUGetDeviceHandle();
  if ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::PrivateUse1 )
  {
    CHECK_TENSOR_IN_DEVICE ( input1 );
    CHECK_TENSOR_IN_DEVICE ( input2 );
    CHECK_TENSOR_IN_DEVICE ( out );
    auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
    auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
    auto output_desc = tpu::TPUGenerateTensorDesc ( out );
    bm_status_t status = sgdnn_binary_cudnn (
                         handle,
                         input1_desc,
                         ADDR_IN_DEVICE ( input1 ),
                         input2_desc,
                         ADDR_IN_DEVICE ( input2 ),
                         output_desc,
                         ADDR_IN_DEVICE ( out ),
                         OP_BINARY_MUL );
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else if ( ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::CPU ) ||
            ( input1.device().type() == DeviceType::CPU && input2.device().type() == DeviceType::PrivateUse1 ) )
  {
    if ( input2.device().type() == DeviceType::CPU )
    {
      TORCH_CHECK ( input2.dim() == 0, "Input2 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input1 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input2.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input2.to ( torch::kFloat );
      }
      else
      {
        Scalar = input2;
      }
      auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input1_desc,
                           ADDR_IN_DEVICE ( input1 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_MUL );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
    else
    {
      TORCH_CHECK ( input1.dim() == 0, "Input1 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input2 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input1.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input1.to ( torch::kFloat );
      }
      else
      {
        Scalar = input1;
      }
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input2_desc,
                           ADDR_IN_DEVICE ( input2 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_MUL );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "mul.out", mul_out_tpu );
}

Tensor & div_out_tpu ( const Tensor & input1,
                       const Tensor & input2,
                       Tensor       & out )
{
#if 0
  auto out_cpu = torch::div ( input1.to ( torch::Device ( "cpu" ) ), input2.to ( torch::Device ( "cpu" ) ) );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto handle = tpu::TPUGetDeviceHandle();
  if ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::PrivateUse1 )
  {
    CHECK_TENSOR_IN_DEVICE ( input1 );
    CHECK_TENSOR_IN_DEVICE ( input2 );
    CHECK_TENSOR_IN_DEVICE ( out );
    auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
    auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
    auto output_desc = tpu::TPUGenerateTensorDesc ( out );
    bm_status_t status = sgdnn_binary_cudnn (
                         handle,
                         input1_desc,
                         ADDR_IN_DEVICE ( input1 ),
                         input2_desc,
                         ADDR_IN_DEVICE ( input2 ),
                         output_desc,
                         ADDR_IN_DEVICE ( out ),
                         OP_BINARY_DIV );
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else if ( ( input1.device().type() == DeviceType::PrivateUse1 && input2.device().type() == DeviceType::CPU ) ||
            ( input1.device().type() == DeviceType::CPU && input2.device().type() == DeviceType::PrivateUse1 ) )
  {
    if ( input2.device().type() == DeviceType::CPU )
    {
      TORCH_CHECK ( input2.dim() == 0, "Input2 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input1 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input2.dtype() == caffe2::TypeMeta::Make<double>() ||
           input2.dtype() == caffe2::TypeMeta::Make<long>() )
      {
        Scalar = input2.to ( torch::kFloat );
      }
      else
      {
        Scalar = input2;
      }
      auto input1_desc = tpu::TPUGenerateTensorDesc ( input1 );
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input1_desc,
                           ADDR_IN_DEVICE ( input1 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_DIV );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
    else
    {
      TORCH_CHECK ( input1.dim() == 0, "Input1 must be a scalar" );
      CHECK_TENSOR_IN_DEVICE ( input2 );
      CHECK_TENSOR_IN_DEVICE ( out );
      Tensor Scalar;
      if ( input1.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        Scalar = input1.to ( torch::kFloat );
      }
      else
      {
        Scalar = input1;
      }
      auto Scalar_desc = tpu::TPUGenerateTensorDesc ( Scalar );
      auto input2_desc = tpu::TPUGenerateTensorDesc ( input2 );
      auto output_desc = tpu::TPUGenerateTensorDesc ( out );
      bm_status_t status = sgdnn_binary_cudnn (
                           handle,
                           input2_desc,
                           ADDR_IN_DEVICE ( input2 ),
                           Scalar_desc,
                           Scalar.data_ptr(),
                           output_desc,
                           ADDR_IN_DEVICE ( out ),
                           OP_BINARY_DIV );
      TORCH_CHECK ( status == BM_SUCCESS );
    }
  }
  else
  {
    TORCH_CHECK ( false, "At least one input is required in TPU device" );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "div.out", div_out_tpu );
}
} // namespace at
