#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"

namespace at
{
Tensor & norm_out_tpu( const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, Tensor & out)
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );

#ifdef SHOW_OP_INFO
  static int count = 0;
  std::cout << "norm_out " << count << std::endl;
  ++count;
#endif
#if 1
  auto out_cpu = norm ( self.cpu(), p,
                         dim, keepdim );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else

#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif

#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::NORM2, timer.ElapsedUS() );
#endif
#endif

  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "norm.out", norm_out_tpu );
}
} // namespace at