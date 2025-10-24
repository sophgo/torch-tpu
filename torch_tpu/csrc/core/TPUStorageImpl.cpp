#include "torch_tpu/csrc/core/TPUStorageImpl.h"

namespace torch_tpu {
/**************************************************************
*****************      TPU Storage Impl       *****************
***************************************************************/
TPUStorageImpl::TPUStorageImpl(
    use_byte_size_t use_byte_size,
    size_t size_bytes,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable) : c10::StorageImpl(
      use_byte_size,
      size_bytes,
      at::DataPtr(std::move(data_ptr)),
      allocator,
      resizable)
{
}

void TPUStorageImpl::release_resources() {
  StorageImpl::release_resources();
}

/**************************************************************
*****************      interface function     *****************
***************************************************************/
c10::intrusive_ptr<c10::StorageImpl> make_tpu_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::Allocator* allocator,
    bool resizable) {
  // Correctly create TPUStorageImpl object.
  c10::intrusive_ptr<c10::StorageImpl> tpu_storage_impl = c10::make_intrusive<TPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size_bytes.as_int_unchecked(),
          allocator->allocate(size_bytes.as_int_unchecked()),
          allocator,
          resizable);
  // There is no need to consider the TPUStorageDesc information, it will be carried out in the subsequent processing.
  return tpu_storage_impl;
}

TPUStorageImpl* GetTpuStorageImpl(c10::StorageImpl* storageImpl) {
  return static_cast<TPUStorageImpl*>(storageImpl);
}

TPUStorageImpl* GetTpuStorageImpl(c10::Storage&& storage) {
  return static_cast<TPUStorageImpl*>(storage.unsafeGetStorageImpl());
}

TPUStorageImpl* GetTpuStorageImpl(const at::Tensor& tensor) {
  return static_cast<TPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl());
}

TPUStorageDesc& GetTpuStorageImplDesc(const at::Tensor &tensor) {
  return static_cast<TPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl())->tpu_desc_;
}

} // namespace torch_npu