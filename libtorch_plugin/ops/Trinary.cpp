#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace at
{

Tensor & addcmul_out_tpu ( const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( tensor1 );
  CHECK_TENSOR_IN_DEVICE ( tensor2 );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = addcmul(self.to(torch::kFloat).cpu(), tensor1.to(torch::kFloat).cpu(), tensor2.to(torch::kFloat).cpu(), value);
  out = out_cpu.to(out.device()).to(out.dtype());
#else
  if ( tpu::TPUIsSameShape(self, tensor1) && tpu::TPUIsSameShape(self, tensor2) )
  {
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnAddCMul (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( tensor1 ),
                       tpu::TPUGenerateSgdnnTensor ( tensor2 ),
                       value.toDouble(),
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::ADDCMUL, timer.ElapsedUS() );
#endif
  }
  else
  {
    LOG( WARNING ) << " addcmulBcast use cpu impl";
    auto out_cpu = addcmul(self.to(torch::kFloat).cpu(), tensor1.to(torch::kFloat).cpu(), tensor2.to(torch::kFloat).cpu(), value);
    out = out_cpu.to(out.device()).to(out.dtype());
    // #ifdef TPU_OP_TIMING
    //   auto timer = tpu::Timer().Start();
    // #endif
    //   bm_status_t status = sgdnnAddCMulBroadCast (
    //                        tpu::TPUGetDeviceHandle(),
    //                        tpu::TPUGenerateSgdnnTensor ( self ),
    //                        tpu::TPUGenerateSgdnnTensor ( tensor1 ),
    //                        tpu::TPUGenerateSgdnnTensor ( tensor2 ),
    //                        value.toDouble(),
    //                        tpu::TPUGenerateSgdnnTensor ( out ) );
    //   TORCH_CHECK ( status == BM_SUCCESS );
    // #ifdef TPU_OP_TIMING
    //   tpu::OpTimer::Instance().AddTime ( tpu::ADDCMULBCast, timer.ElapsedUS() );
    // #endif 
  }
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
  bm_status_t status = sgdnnAddCDiv (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( tensor1 ),
                       tpu::TPUGenerateSgdnnTensor ( tensor2 ),
                       value.toDouble(),
                       tpu::TPUGenerateSgdnnTensor ( out ) );
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
