#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/native/Resize.h>

#include "TPUTorchUtils.h"
#include "Resize.h"

#include "common/config.h"

namespace at {
namespace native{

template<typename T>
TensorImpl* resize_impl_tpu_( TensorImpl* self, ArrayRef<T> size, at::OptionalArrayRef<T> stride, bool resize_storage)
{
  if (self->generic_sizes<T>() == size && (!stride || self->generic_strides<T>() == stride.value())) {
    return self;
  }

  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->generic_storage_offset<T>();
  T storage_size = T(1);
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    self->generic_set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }

  if (resize_storage) {
    maybe_resize_storage_tpu(self, std::move(storage_size));
  }

  return self;
}

} // namespace native

Tensor& set_tpu_(Tensor & result, Storage source)
{
  SHOW_TENSOR_OP(result);
  CHECK_TENSOR_IN_DEVICE(result);
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(std::move(source), 0, new_size, {});
}

Tensor& set_storage_tpu_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride)
{
  native::checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  native::resize_impl_tpu_(result.unsafeGetTensorImpl(), size, stride_opt, false);
  SHOW_TENSOR_OP(result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("set_.source_Storage", set_tpu_);
  m.impl("set_.source_Storage_storage_offset", set_storage_tpu_);
}
} //namespace at