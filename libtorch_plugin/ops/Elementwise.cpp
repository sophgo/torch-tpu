#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

//#define TPU_OP_TIMING

namespace at
{
Tensor & sqrt_out_tpu ( const Tensor & self, Tensor & out  )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnSqrt (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::SQRT, timer.ElapsedUS() );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "sqrt.out", sqrt_out_tpu );
}
} // namespace at


namespace at
{
Tensor & rsqrt_out_tpu ( const Tensor & self, Tensor & out  )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnRsqrt (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::RSQRT, timer.ElapsedUS() );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "rsqrt.out", rsqrt_out_tpu );
}
} // namespace at

