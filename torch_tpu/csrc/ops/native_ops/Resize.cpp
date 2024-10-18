#include "Resize.h"
#include "TPUTorchUtils.h"

namespace at::native {

void resize_bytes_tpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");

  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_nbytes(0);
    return;
  }

  c10_tpu::TPUGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);

  at::DataPtr new_data;
  if (size_bytes != 0) {
    new_data = storage->allocator()->allocate(size_bytes);
  }
  at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
  const auto old_capacity = storage->nbytes();
  storage->set_nbytes(size_bytes);
  const auto copy_capacity = std::min(size_bytes, old_capacity);
  if (old_data != nullptr && copy_capacity > 0) {
    tpu::TPUCopyDeviceToDevice(storage->mutable_data(), old_data.get(),
                                   copy_capacity, true);
  }
}

} // namespace at::native {
