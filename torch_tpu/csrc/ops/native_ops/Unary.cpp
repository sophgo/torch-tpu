#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor &bitwise_not_out_tpu(const Tensor &self, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  if (self.dim() == 0) {
    auto out_cpu = bitwise_not(self.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (IS_TPU_TENSOR(self)) {
    TIMING_START;
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnBitwiseNotAsync(
                  stream,
                  tpu::TPUGenerateTpudnnTensor ( stream,self ),
                  tpu::TPUGenerateTpudnnTensor ( stream,out ) );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS );

    TIMING_END(tpu::BITWISE_NOT);
  }
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}

Tensor bitwise_not_tpu(const Tensor &self) {
  auto out = empty_like(self);
  return bitwise_not_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_not", bitwise_not_tpu);
  m.impl("bitwise_not.out", bitwise_not_out_tpu);
}

Tensor &cbrt_out_tpu(const Tensor &self, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  if (self.dim() == 0) {
    auto out_cpu = cbrt(self.item().toFloat());
    tpu::TPUCopyHostToDevice(out.data_ptr(), &out_cpu, out.nbytes());
  } else if (IS_TPU_TENSOR(self)) {
    TIMING_START;
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnCbrtAsync(
                  stream,
                  tpu::TPUGenerateTpudnnTensor ( stream,self ),
                  tpu::TPUGenerateTpudnnTensor ( stream,out ) );
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS );
    TIMING_END(tpu::CBRT);
  }
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
Tensor cbrt_tpu(const Tensor &self) {
  auto out = empty_like(self);
  return cbrt_out_tpu(self, out);
}
// TORCH_LIBRARY_IMPL ( aten, TPU, m )
// {
//   m.impl ( "bitwise_not", cbrt_tpu );    //
//   pytorch2.0.1版本库没有cbrt接口，借用bitwise_not接口实现 m.impl (
//   "bitwise_not.out", cbrt_out_tpu );
// }
// TORCH_LIBRARY_IMPL ( aten, TPU, m )     // 换成pytorch2.1以上版本后可使用？
// {
//   m.impl ( "cbrt", cbrt_tpu );
//   m.impl ( "cbrt.out", cbrt_out_tpu );
// }

} // namespace at
