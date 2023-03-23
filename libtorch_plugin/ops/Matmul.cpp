#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
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
  TORCH_CHECK ( alpha.toDouble() == 1. );
  TORCH_CHECK ( beta.toDouble() == 1. );
  auto mat1_mat2 = mm ( mat1, mat2 );
  add_out ( out, self, mat1_mat2, 1. );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
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
  auto status = sgdnn_general_matmul (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateTensorDesc ( self ),
                ADDR_IN_DEVICE ( self ),
                tpu::TPUGenerateTensorDesc ( mat2 ),
                ADDR_IN_DEVICE ( mat2 ),
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                false );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::MM, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
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
  auto status = sgdnn_batch_matmul (
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateTensorDesc ( self ),
                ADDR_IN_DEVICE ( self ),
                tpu::TPUGenerateTensorDesc ( mat2 ),
                ADDR_IN_DEVICE ( mat2 ),
                tpu::TPUGenerateTensorDesc ( out ),
                ADDR_IN_DEVICE ( out ),
                false,
                false );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::BMM, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "bmm.out", bmm_out_tpu );
}

} // namespace at
