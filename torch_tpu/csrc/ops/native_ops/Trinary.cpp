#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at {

Tensor &addcmul_out_tpu(const Tensor &self, const Tensor &tensor1,
                        const Tensor &tensor2, const Scalar &value,
                        Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(tensor1);
  CHECK_TENSOR_IN_DEVICE(tensor2);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = addcmul(self.to(torch::kFloat).cpu(), tensor1.to(torch::kFloat).cpu(), tensor2.to(torch::kFloat).cpu(), value);
  out = out_cpu.to(out.device()).to(out.dtype());
#else
  if (tpu::TPUIsSameShape(self, tensor1) &&
      tpu::TPUIsSameShape(self, tensor2)) {
    TIMING_START;

    auto status = sgdnnAddCMul(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        tpu::TPUGenerateSgdnnTensor(tensor1),
        tpu::TPUGenerateSgdnnTensor(tensor2), value.toDouble(),
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::ADDCMUL);
  } else {
    auto self_t = tpu::TPUGenerateSgdnnTensor(self);
    auto tensor1_t = tpu::TPUGenerateSgdnnTensor(tensor1);
    auto tensor2_t = tpu::TPUGenerateSgdnnTensor(tensor2);
    int maxdim = self_t.dim > tensor1_t.dim
                     ? self_t.dim > tensor2_t.dim ? self_t.dim : tensor2_t.dim
                 : tensor1_t.dim > tensor2_t.dim ? tensor1_t.dim
                                                 : tensor2_t.dim;
    if (self_t.dim != maxdim) {
      int stride = 1;
      for (int i = 0; i < maxdim; i++) {
        if (i < self_t.dim) {
          self_t.shape[maxdim - i - 1] = self_t.shape[self_t.dim - i - 1];
          self_t.stride[maxdim - i - 1] = stride;
        } else {
          self_t.shape[maxdim - i - 1] = 1;
          self_t.stride[maxdim - i - 1] = stride;
        }
        stride *= self_t.shape[maxdim - i - 1];
      }
      self_t.dim = maxdim;
    }

    if (tensor1_t.dim != maxdim) {
      int stride = 1;
      for (int i = 0; i < maxdim; i++) {
        if (i < tensor1_t.dim) {
          tensor1_t.shape[maxdim - i - 1] =
              tensor1_t.shape[tensor1_t.dim - i - 1];
          tensor1_t.stride[maxdim - i - 1] = stride;
        } else {
          tensor1_t.shape[maxdim - i - 1] = 1;
          tensor1_t.stride[maxdim - i - 1] = stride;
        }
        stride *= tensor1_t.shape[maxdim - i - 1];
      }
      tensor1_t.dim = maxdim;
    }

    if (tensor2_t.dim != maxdim) {
      int stride = 1;
      for (int i = 0; i < maxdim; i++) {
        if (i < tensor2_t.dim) {
          tensor2_t.shape[maxdim - i - 1] =
              tensor2_t.shape[tensor2_t.dim - i - 1];
          tensor2_t.stride[maxdim - i - 1] = stride;
        } else {
          tensor2_t.shape[maxdim - i - 1] = 1;
          tensor2_t.stride[maxdim - i - 1] = stride;
        }
        stride *= tensor2_t.shape[maxdim - i - 1];
      }
      tensor2_t.dim = maxdim;
    }

    TIMING_START;

    auto status = sgdnnAddCMulBcast(tpu::TPUGetDeviceResource(), self_t,
                                    tensor1_t, tensor2_t, value.toDouble(),
                                    tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::ADDCMUL);
  }
#endif
  SHOW_TENSOR_OP(self, tensor1, tensor2, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("addcmul.out", addcmul_out_tpu); }

Tensor &addcdiv_out_tpu(const Tensor &self, const Tensor &tensor1,
                        const Tensor &tensor2, const Scalar &value,
                        Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(tensor1);
  CHECK_TENSOR_IN_DEVICE(tensor2);
  CHECK_TENSOR_IN_DEVICE(out);
  TIMING_START;
  auto status = sgdnnAddCDiv(
      tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(tensor1),
      tpu::TPUGenerateSgdnnTensor(tensor2), value.toDouble(),
      tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == SG_SUCCESS);
  TIMING_END(tpu::ADDCDIV);
  SHOW_TENSOR_OP(self, tensor1, tensor2, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("addcdiv.out", addcdiv_out_tpu); }

} // namespace at
