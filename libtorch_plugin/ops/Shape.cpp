#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

#define CONSTANT (0)
#define REFLECT (1)
#define SYMMETRIC (2)
#define REPLICATE (3)
#define CIRCULAR (4)

namespace at {
//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
Tensor alias_with_sizes_and_strides(const Tensor &self, const Vec &sizes,
                                    const Vec &strides) {
  // caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  Tensor self_;
  if (self.is_quantized()) {
    self_ = at::detail::make_tensor<QTensorImpl>(
        c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(),
        self.dtype(), get_qtensorimpl(self)->quantizer());
    auto *self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  } else {
    self_ = at::detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW,
                                                Storage(self.storage()),
                                                self.key_set(), self.dtype());
    auto *self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
  }
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor view_tpu(const Tensor &self, c10::IntArrayRef size) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("view", view_tpu); }

Tensor _reshape_alias_tpu(const Tensor &input, IntArrayRef sizes,
                          IntArrayRef strides) {
  return alias_with_sizes_and_strides(input, sizes, strides);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("_reshape_alias", _reshape_alias_tpu);
}

Tensor as_strided_tpu(const Tensor &self, IntArrayRef size, IntArrayRef stride,
                      c10::optional<int64_t> storage_offset) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  Tensor out;
  out = at::detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW,
                                            Storage(self.storage()),
                                            self.key_set(), self.dtype());
  auto *self_tmp_ = out.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(storage_offset.has_value()
                                    ? storage_offset.value()
                                    : self.storage_offset());
  self_tmp_->set_sizes_and_strides(size, stride);
  namedinference::propagate_names(out, self);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("as_strided", as_strided_tpu); }

Tensor reshape_tpu(const Tensor &self, IntArrayRef proposed_shape) {
  CHECK_TENSOR_IN_DEVICE(self);
  at::DimVector shape = at::infer_size_dv(proposed_shape, self.numel());
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), shape);
  return alias_with_sizes_and_strides(self, shape, *stride);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("reshape_symint", reshape_tpu); }

Tensor &expand_out_tpu(const Tensor &self, const IntArrayRef output_size,
                       Tensor &out) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(out);
  TORCH_CHECK(self.dim() > 0 && self.dim() <= 4,
              "The expand supports up to 4d tensors");
  TORCH_CHECK(
      static_cast<size_t>(self.dim()) <= output_size.size(),
      "The number of sizes provided (", output_size.size(),
      ") must be greater or equal to the number of dimensions in the tensor (",
      self.dim(), ").");

  std::vector<int64_t> repeat_size = std::vector<int64_t>(output_size.size());
  std::vector<int64_t> input_size = self.sizes().vec();
  int in_idx = input_size.size() - 1;
  for (int i = output_size.size() - 1; i >= 0; --i) {
    if (in_idx >= 0) {
      TORCH_CHECK(input_size[in_idx] == output_size[i] ||
                      input_size[in_idx] == 1 || output_size[i] == -1,
                  "The expanded size of the tensor (", output_size[i],
                  ") must match the existing size (", input_size[in_idx],
                  ") at non-singleton dimension ", i);

      if (input_size[in_idx] == output_size[i] || output_size[i] == -1) {
        repeat_size[i] = 1;
      } else if (input_size[in_idx] == 1) {
        repeat_size[i] = output_size[i];
      }
      --in_idx;
    } else {
      TORCH_CHECK(output_size[i] != -1,
                  "The expanded size of the tensor (-1) is not "
                  "allowed in a leading, non-existing dimension 0.");

      repeat_size[i] = output_size[i];
    }
  }

#if 0
  // repeat not implemented in TPU now, use cpu
  auto self_cpu = self.cpu().repeat(repeat_size);
  tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(),
                           out.nbytes());
#else
  TIMING_START
  out = self.repeat(repeat_size);
  TIMING_END(tpu::EXPAND)
#endif
  return out;
}

Tensor expand_tpu(const Tensor &self, const IntArrayRef output_size,
                  bool implicit = false) {
  auto out = empty(output_size, self.options());
  return expand_out_tpu(self, output_size, out);
}

// TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("expand", expand_tpu); }

Tensor constant_pad_nd_tpu(const Tensor &self, const IntArrayRef padding,
                           const Scalar value) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }

  TORCH_CHECK((padding.size() % 2) == 0, "Length of pad must be even.");
  TORCH_CHECK(self.dim() >= (padding.size() / 2),
              "Length of pad should be no more than twice the number of "
              "dimensions of the input.");
  TORCH_CHECK(self.dim() <= 4,
              "The dim of the tensor should be less than or equal to 4");

  Tensor out;
  if (self.dim() == 0 || padding.size() == 0) {
    out = Tensor(self);
    return out;
  }

  std::vector<int64_t> size_vec(self.dim());
  std::vector<int> pad_vec(padding.begin(), padding.end());
  int cur_shape, pad_size = padding.size();
  for (int i = self.dim() - 1, j = 0; i >= 0; i--, j += 2) {
    cur_shape = self.size(i);
    if (j < pad_size) {
      cur_shape += padding[j] + padding[j + 1];
    }
    size_vec[i] = cur_shape;
  }

  IntArrayRef size(size_vec);
  out = empty(size, self.options());
  TIMING_START
  // 0 for constant pad
  bm_status_t status =
      sgdnnPad(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
               pad_vec.data(), pad_vec.size(), value.toFloat(), CONSTANT, false,
               tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::CONSTANT_PAD)

  return out;
}
// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("constant_pad_nd", constant_pad_nd_tpu);
// }

Tensor &reflection_pad2d_out_tpu(const Tensor &self, IntArrayRef padding,
                                 Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  if (self.dim() == 0) {
    auto out_cpu = reflection_pad2d(self.cpu(), padding);
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_cpu.data_ptr(),
                               out.nbytes());
    return out;
  }
  std::vector<int> pad(padding.begin(), padding.end());

  TIMING_START;
  bm_status_t status = sgdnnPad(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self), pad.data(),
      pad.size(), 0, REFLECT, false, tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REFLECTION_PAD2D);
#endif
  return out;
}

Tensor reflection_pad2d_tpu(const Tensor &self, IntArrayRef padding) {
  CHECK_TENSOR_IN_DEVICE(self);
  TORCH_CHECK(self.dim() >= 2);
  std::vector<int64_t> size_vec(self.sizes().begin(), self.sizes().end());
  size_vec[self.dim() - 2] += padding[2] + padding[3];
  size_vec[self.dim() - 1] += padding[0] + padding[1];
  IntArrayRef size(size_vec.data(), size_vec.size());
  auto out = empty(size, self.options());
  return reflection_pad2d_out_tpu(self, padding, out);
}

// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("reflection_pad2d", reflection_pad2d_tpu);
//   m.impl("reflection_pad2d.out", reflection_pad2d_out_tpu);
// }

Tensor &replication_pad2d_out_tpu(const Tensor &self, IntArrayRef padding,
                                  Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  if (self.dim() == 0) {
    auto out_cpu = replication_pad2d(self.cpu(), padding);
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_cpu.data_ptr(),
                               out.nbytes());
    return out;
  }
  std::vector<int> pad(padding.begin(), padding.end());

  TIMING_START;
  bm_status_t status = sgdnnPad(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self), pad.data(),
      pad.size(), 0, REPLICATE, false, tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REPLICATION_PAD2D);
#endif
  return out;
}

// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("replication_pad2d.out", replication_pad2d_out_tpu);
// }

Tensor &replication_pad3d_out_tpu(const Tensor &self, IntArrayRef padding,
                                  Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  if (self.dim() == 0) {
    auto out_cpu = replication_pad3d(self.cpu(), padding);
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_cpu.data_ptr(),
                               out.nbytes());
    return out;
  }
  std::vector<int> pad(padding.begin(), padding.end());

  TIMING_START;
  bm_status_t status = sgdnnPad(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self), pad.data(),
      pad.size(), 0, REPLICATE, true, tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REPLICATION_PAD3D);
#endif
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("replication_pad3d.out", replication_pad3d_out_tpu);
}

Tensor clone_preserve_strides(const at::Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.has_storage());
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    return self.clone();
  }
  auto dtype_size = self.dtype().itemsize();
  auto nbytes = self.storage().sym_nbytes();
  TORCH_INTERNAL_ASSERT(nbytes % dtype_size == 0);
  auto numel = nbytes / dtype_size;
  auto self_full_size = self.as_strided_symint({std::move(numel)}, {1}, 0);
  auto clone = self_full_size.clone();
  auto out = clone.as_strided_symint(self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());
  return out;
}

Tensor as_strided_scatter_tpu(const Tensor &self, const Tensor &src, IntArrayRef size, IntArrayRef stride,
                      c10::optional<int64_t> storage_offset) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(src);
  Tensor out = clone_preserve_strides(self);
  Tensor slice = as_strided(out, size, stride, std::move(storage_offset));
  TORCH_CHECK(slice.sym_sizes() == src.sym_sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sym_sizes(), "slice size = ", slice.sym_sizes());
  slice.copy_(src);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("as_strided_scatter", as_strided_scatter_tpu); }

}  // namespace at
