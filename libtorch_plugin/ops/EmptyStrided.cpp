#include <torch/library.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUAllocator.h>
#include <TPUDeviceManager.h>

#include "common/config.h"

namespace at
{
Tensor empty_strided_tpu ( IntArrayRef                size,
                           IntArrayRef                stride,
                           c10::optional<ScalarType>  dtype_opt,
                           c10::optional<Layout>      layout_opt,
                           c10::optional<Device>      device_opt,
                           c10::optional<bool>        pin_memory_opt )
{
  if ( device_opt.has_value() )
  {
    tpu::TPUSetDeviceIndex ( device_opt.value().index() );
  }
  auto scalar_type = dtype_or_default ( dtype_opt );
  auto pin_memory = pinned_memory_or_default ( pin_memory_opt );
  at::detail::check_size_nonnegative ( size );
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta ( scalar_type );
  auto size_bytes = at::detail::computeStorageNbytes ( size, stride, dtype.itemsize() );
  auto allocator = c10::GetTPUAllocator();
  auto storage_impl = c10::make_intrusive<StorageImpl> (
                      c10::StorageImpl::use_byte_size_t(),
                      size_bytes,
                      allocator,
                      /*resizeable=*/true );
  constexpr c10::DispatchKeySet ks ( c10::DispatchKey::TPU );
  auto tensor = detail::make_tensor_base<TensorImpl> ( std::move ( storage_impl ), ks, dtype );
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides ( size, stride );
  TORCH_CHECK ( tensor.is_contiguous() );
  return tensor;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "empty_strided", empty_strided_tpu );
}

Tensor empty_memory_format_tpu (
IntArrayRef                       size,
c10::optional<ScalarType>         dtype_opt,
c10::optional<Layout>             layout_opt,
c10::optional<Device>             device_opt,
c10::optional<bool>               pin_memory_opt,
c10::optional<c10::MemoryFormat>  memory_format_opt )
{
  if ( device_opt.has_value() )
  {
    tpu::TPUSetDeviceIndex ( device_opt.value().index() );
  }
  auto scalar_type = dtype_or_default ( dtype_opt );
  auto pin_memory = pinned_memory_or_default ( pin_memory_opt );
  constexpr c10::DispatchKeySet ks ( c10::DispatchKey::TPU );
  auto allocator = c10::GetTPUAllocator();
  at::detail::check_size_nonnegative ( size );
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta ( scalar_type );
  size_t size_bytes = at::detail::computeStorageNbytesContiguous ( size, dtype.itemsize() );
  auto storage_impl = c10::make_intrusive<StorageImpl> (
                      c10::StorageImpl::use_byte_size_t(),
                      size_bytes,
                      allocator->allocate ( size_bytes ),
                      allocator,
                      /*resizeable=*/true );
  auto tensor = detail::make_tensor_base<TensorImpl> ( std::move ( storage_impl ), ks, dtype );
  // Default TensorImpl has size [0]
  if ( size.size() != 1 || size[0] != 0 )
  {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous ( size );
  }
  if ( memory_format_opt.has_value() )
  {
    // Restriding a just-created empty contiguous tensor does nothing.
    if ( *memory_format_opt != MemoryFormat::Contiguous )
    {
      LOG ( FATAL );
      tensor.unsafeGetTensorImpl()->empty_tensor_restride ( *memory_format_opt );
    }
  }
  TORCH_CHECK ( tensor.is_contiguous() );
  return tensor;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "empty.memory_format", empty_memory_format_tpu );
}
} // namespace at
