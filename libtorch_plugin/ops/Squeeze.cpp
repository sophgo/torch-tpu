#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor &squeeze_out_tpu(const Tensor &self, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

  auto self_cpu = squeeze ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
  if (self.dim() == 0) {
    auto self_cpu = squeeze(self.cpu());
    tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(),
                             self.nbytes());
  } else if (IS_TPU_TENSOR(self)) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnSqueeze(tpu::TPUGetDeviceHandle(),
                                      tpu::TPUGenerateSgdnnTensor(self),
                                      tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::SQUEEZE, timer.ElapsedUS());
#endif
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
}

Tensor squeeze_tpu(const Tensor &self) {
  auto in_shape = self.sizes();
  auto in_dims = self.dim();

  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_dims; ++i) {
    if (in_shape[i] != 1) {
      out_shape.push_back(in_shape[i]);
    }
  }

  at::IntArrayRef output_shape_ref(out_shape);
  auto out = empty(output_shape_ref, self.options());
  return squeeze_out_tpu(self, out);
}

Tensor squeeze_dim_tpu(const Tensor &self, const int64_t dim) {
  auto in_shape = self.sizes();
  auto in_dims = self.dim();
  TORCH_CHECK(0 <= dim && dim < in_dims,
              "Dimension out of range (expected to "
              "be in range of [0 ",
              in_dims - 1, "], but got ", dim, ")");
  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_dims; ++i) {
    if (i == dim && in_shape[i] == 1)
      continue;
    out_shape.push_back(in_shape[i]);
  }

  at::IntArrayRef output_shape_ref(out_shape);
  auto out = empty(output_shape_ref, self.options());
  return squeeze_out_tpu(self, out);
}

Tensor squeeze_dims_tpu(const Tensor &self, const c10::ArrayRef<long> dims) {
  auto in_shape = self.sizes();
  auto in_dims = self.dim();

  std::set<int64_t> dim_set;
  for (auto dim : dims) {
    TORCH_CHECK(0 <= dim && dim < in_dims,
                "Dimension out of range (expected to "
                "be in range of [0 ",
                in_dims - 1, "], but got ", dim, ")");
    dim_set.insert(dim);
  }

  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_dims; ++i) {
    if (dim_set.count(i) && in_shape[i] == 1)
      continue;
    out_shape.push_back(in_shape[i]);
  }

  at::IntArrayRef output_shape_ref(out_shape);
  auto out = empty(output_shape_ref, self.options());
  return squeeze_out_tpu(self, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("squeeze", squeeze_tpu);
  m.impl("squeeze.dim", squeeze_dim_tpu);
  m.impl("squeeze.dims", squeeze_dims_tpu);
}

} // namespace at