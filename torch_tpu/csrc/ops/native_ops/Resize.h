#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorUtils.h>

#include "TPUAllocator.h"

#include <utility>

namespace at::native {

static inline void resize_bytes_tpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");

  if (size_bytes == 0) {
    storage->set_nbytes(0);
    return;
  }

  c10::Device device = storage->device();
  c10_tpu::TPUGuard guard(device.index());
  const auto old_capacity = storage->nbytes();
  at::DataPtr new_data;
  if (size_bytes != 0) {
    new_data = storage->allocator()->allocate(size_bytes);
  }
  const at::DataPtr& old_data = storage->data_ptr();
  const auto copy_capacity = std::min(size_bytes, old_capacity);
  if (old_data != nullptr && copy_capacity > 0) {
    tpu::TPUCopyDeviceToDevice(new_data.get(), old_data.get(), copy_capacity, false);
  }
  storage->set_data_ptr_noswap(std::move(new_data));
  storage->set_nbytes(size_bytes);
}

static inline void maybe_resize_storage_tpu(TensorImpl* self, size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in cuda/Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->unsafe_storage();
  if (!storage) {
    auto new_storage = c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        new_size_bytes,
        c10_tpu::GetTPUAllocator(),
        true);
    self->set_storage_keep_dtype(std::move(new_storage));
  } else if (new_size_bytes > storage.nbytes()) {
    resize_bytes_tpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

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
} // namespace at::native