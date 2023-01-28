#include <torch/library.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUAllocator.h>

namespace at
{
Tensor empty_strided_tpu ( IntArrayRef size,
                           IntArrayRef stride,
                           c10::optional<ScalarType> dtype_opt,
                           c10::optional<Layout> layout_opt,
                           c10::optional<Device> device_opt,
                           c10::optional<bool> pin_memory_opt )
{
  auto scalar_type = dtype_or_default ( dtype_opt );
  auto pin_memory = pinned_memory_or_default ( pin_memory_opt );
  at::detail::check_size_nonnegative ( size );
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta ( scalar_type );
  auto size_bytes = at::detail::computeStorageNbytes (
                    size, stride, dtype.itemsize() );
  auto allocator = c10::GetTPUAllocator();
  auto storage_impl = c10::make_intrusive<StorageImpl> (
                      c10::StorageImpl::use_byte_size_t(),
                      size_bytes,
                      allocator,
                      /*resizeable=*/true );
  constexpr c10::DispatchKeySet ks ( c10::DispatchKey::PrivateUse1 );
  auto tensor = detail::make_tensor_base<TensorImpl> (
                std::move ( storage_impl ), ks, dtype );
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides ( size, stride );
  return tensor;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "empty_strided", empty_strided_tpu );
}
} // namespace at
