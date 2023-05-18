#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <TPUModule.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING
//#define SHOW_OP_INFO

namespace at
{
Tensor & addmm_out_tpu ( const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar & beta, const Scalar & alpha, Tensor & out )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "Addmm " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mat1 );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = addmm ( self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (( alpha.toDouble() == 1. ) && ( beta.toDouble() == 1. ))
  {
    linear_out( out, mat1, mat2, self);
  }
  else
  {
    TORCH_CHECK( false );
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addmm.out", addmm_out_tpu );
}

Tensor & mm_out_tpu ( const Tensor & self, const Tensor & mat2, Tensor & out )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "mm " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = mm ( self.cpu(), mat2.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto computeType = SG_DTYPE_FP32;
  auto accuracy = tpu::GetMatrixMultiplyAccuracy();
  if ( self.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    if ( accuracy == tpu::ALGORITHM_ACCURACY_FP32 )
    {
      computeType = SG_DTYPE_FP32;
    }
    else if ( accuracy == tpu::ALGORITHM_ACCURACY_FP16 )
    {
      computeType = SG_DTYPE_FP16;
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported Matmul Accuracy" );
    }
  }
  else if ( self.dtype() == caffe2::TypeMeta::Make<at::Half>() )
  {
    computeType = SG_DTYPE_FP16;
  }
  auto status = sgdnn_general_matmul (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateTensorDesc ( self ),
                ADDR_IN_DEVICE ( self ),
                tpu::TPUGenerateTensorDesc ( mat2 ),
                ADDR_IN_DEVICE ( mat2 ),
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                false,
                computeType );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::MM, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mm.out", mm_out_tpu );
}

Tensor & bmm_out_tpu ( const Tensor & self, const Tensor & mat2, Tensor & out )
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "bmm " << count << std::endl;
  ++count;
#endif
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = bmm ( self.cpu(), mat2.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto computeType = SG_DTYPE_FP32;
  auto accuracy = tpu::GetMatrixMultiplyAccuracy();
  if ( self.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    if ( accuracy == tpu::ALGORITHM_ACCURACY_FP32 )
    {
      computeType = SG_DTYPE_FP32;
    }
    else if ( accuracy == tpu::ALGORITHM_ACCURACY_FP16 )
    {
      computeType = SG_DTYPE_FP16;
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported Matmul Accuracy" );
    }
  }
  else if ( self.dtype() == caffe2::TypeMeta::Make<at::Half>() )
  {
    computeType = SG_DTYPE_FP16;
  }
  auto status = sgdnn_batch_matmul (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateTensorDesc ( self ),
                ADDR_IN_DEVICE ( self ),
                tpu::TPUGenerateTensorDesc ( mat2 ),
                ADDR_IN_DEVICE ( mat2 ),
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                false,
                false,
                computeType );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BMM, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "bmm.out", bmm_out_tpu );
}

Tensor & linear_out_tpu ( const Tensor & self, const Tensor & mat2, const c10::optional<Tensor> & bias_opt, Tensor & out)
{
  static int count = 0;
#ifdef SHOW_OP_INFO
  std::cout << "linear " << count << std::endl;
  ++count;
#endif
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor ( bias_opt );
  const Tensor & bias = *bias_maybe_owned;
  if ( bias.defined() ) { CHECK_TENSOR_IN_DEVICE ( bias ); }
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mat2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = linear ( self.cpu(), mat2.cpu(), bias.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto computeType = SG_DTYPE_FP32;
  auto accuracy = tpu::GetMatrixMultiplyAccuracy();
  if ( self.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    if ( accuracy == tpu::ALGORITHM_ACCURACY_FP32 )
    {
      computeType = SG_DTYPE_FP32;
    }
    else if ( accuracy == tpu::ALGORITHM_ACCURACY_FP16 )
    {
      computeType = SG_DTYPE_FP16;
    }
    else
    {
      TORCH_CHECK ( false, "Unsupported Matmul Accuracy" );
    }
  }
  else if ( self.dtype() == caffe2::TypeMeta::Make<at::Half>() )
  {
    computeType = SG_DTYPE_FP16;
  }
  auto status = sgdnn_linear (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateTensorDesc ( self ),
                ADDR_IN_DEVICE ( self ),
                tpu::TPUGenerateTensorDesc ( mat2 ),
                ADDR_IN_DEVICE ( mat2 ),
                tpu::TPUGenerateTensorDesc ( bias ),
                bias.defined() ? ADDR_IN_DEVICE ( bias ) : nullptr,
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                false,
                computeType );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::LINEAR, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "linear.out", linear_out_tpu );
}
} // namespace at
