#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <TPUModule.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at
{
Tensor index_select_tpu ( const Tensor & self, int64_t dim, const Tensor & index )
{
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  CHECK_TENSOR_IN_DEVICE ( self );
  auto out_cpu = index_select ( self.cpu(), dim, index.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::INDEX_SELECT, timer.ElapsedUS() );
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "index_select", index_select_tpu );
}
} // namespace at
