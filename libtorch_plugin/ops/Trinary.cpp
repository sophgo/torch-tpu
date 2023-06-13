#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at
{

Tensor & addcmul_out_tpu ( const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( tensor1 );
  CHECK_TENSOR_IN_DEVICE ( tensor2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_addcmul (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( tensor1 ),
                       ADDR_IN_DEVICE ( tensor1 ),
                       tpu::TPUGenerateTensorDesc ( tensor2 ),
                       ADDR_IN_DEVICE ( tensor2 ),
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ),
                       value.toDouble() );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::ADDCMUL, timer.ElapsedUS() );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addcmul.out", addcmul_out_tpu );
}

Tensor & addcdiv_out_tpu ( const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( tensor1 );
  CHECK_TENSOR_IN_DEVICE ( tensor2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnn_addcdiv (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( tensor1 ),
                       ADDR_IN_DEVICE ( tensor1 ),
                       tpu::TPUGenerateTensorDesc ( tensor2 ),
                       ADDR_IN_DEVICE ( tensor2 ),
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ),
                       value.toDouble() );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::ADDCDIV, timer.ElapsedUS() );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "addcdiv.out", addcdiv_out_tpu );
}

} // namespace at
