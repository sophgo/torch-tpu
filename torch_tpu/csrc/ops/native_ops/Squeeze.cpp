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
    return out;
#else
  if (self.dim() == 0) {
    auto self_cpu = squeeze(self.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  } else if (IS_TPU_TENSOR(self)) {
    auto in_shape = self.sizes();
    for (int i = 0; i < self.dim(); ++i) {
      if (in_shape[i] == 0) {
        auto self_cpu = squeeze(self.cpu());
        tpu::TPUCopyHostToDevice(
            out.data_ptr(), self_cpu.contiguous().data_ptr(), out.nbytes());
        return out;
      }
    }
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

Tensor &unsqueeze_out_tpu(const Tensor &self, Tensor &out, int64_t &dim) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
    return out;
#else
  if (self.dim() == 0) {
    auto self_cpu = unsqueeze(self.cpu(), dim);
    tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  } else if (IS_TPU_TENSOR(self)) {
    auto in_shape = self.sizes();
    for (int i = 0; i < self.dim(); ++i) {
      if (in_shape[i] == 0) {
        auto self_cpu = unsqueeze(self.cpu(), dim);
        tpu::TPUCopyHostToDevice(
            out.data_ptr(), self_cpu.contiguous().data_ptr(), out.nbytes());
        return out;
      }
    }
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // same as squeeze
    bm_status_t status = sgdnnSqueeze(tpu::TPUGetDeviceHandle(),
                                      tpu::TPUGenerateSgdnnTensor(self),
                                      tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::UNSQUEEZE, timer.ElapsedUS());
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

Tensor squeeze_dim_tpu(const Tensor &self, int64_t dim) {
  auto in_shape = self.sizes();
  auto in_dims = self.dim();
  TORCH_CHECK(-in_dims <= dim && dim < in_dims,
              "Dimension out of range (expected to "
              "be in range of [",
              -in_dims, " ,", in_dims - 1, "], but got ", dim, ")");
  if (dim < 0)
    dim += in_dims;
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

Tensor squeeze_dims_tpu(const Tensor &self, c10::ArrayRef<long> dims) {
  auto in_shape = self.sizes();
  auto in_dims = self.dim();

  std::set<int64_t> dim_set;
  for (auto dim : dims) {
    TORCH_CHECK(-in_dims <= dim && dim < in_dims,
                "Dimension out of range (expected to "
                "be in range of [",
                -in_dims, " ,", in_dims - 1, "], but got ", dim, ")");
    if (dim < 0)
      dim += in_dims;
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

Tensor unsqueeze_tpu(const Tensor &self, int64_t dim) {
  auto in_shape = self.sizes();
  auto in_dims = self.dim();

  TORCH_CHECK(-in_dims - 1 <= dim && dim <= in_dims,
              "Dimension out of range (expected to "
              "be in range of [",
              -in_dims - 1, " ,", in_dims, "], but got ", dim, ")");
  if (dim < 0)
    dim += (in_dims + 1);
  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_dims; ++i) {
    if (i == dim)
      out_shape.push_back(1);
    out_shape.push_back(in_shape[i]);
  }
  if (dim == in_dims)
    out_shape.push_back(1);
  at::IntArrayRef output_shape_ref(out_shape);
  auto out = empty(output_shape_ref, self.options());
  return unsqueeze_out_tpu(self, out, dim);
}

// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("squeeze", squeeze_tpu);
//   m.impl("squeeze.dim", squeeze_dim_tpu);
//   m.impl("squeeze.dims", squeeze_dims_tpu);
//   m.impl("unsqueeze", unsqueeze_tpu);
// }

} // namespace at