#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "TPUTorchUtils.h"

namespace at
{
Tensor & neg_out_tpu ( const Tensor & self, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( self );
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( out );
#if 0
  auto out_cpu = neg( self.cpu());
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (!self.is_contiguous() || !out.is_contiguous()) {
    if (out.is_contiguous()) {
      out = neg(self.contiguous());
    } else {
      auto out_ = neg(self.contiguous());
      TIMING_START;
      sgdnnStridedCopy(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
      TIMING_END(tpu::STRIDED_COPY);
    }
    SHOW_TENSOR_OP(self, out);
    return out;
  }
  TIMING_START;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnNegAsync(
              stream,
              tpu::TPUGenerateTpudnnTensor(stream, self),
              tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END ( tpu::NEG );
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("neg.out", neg_out_tpu); }
} // namespace at
