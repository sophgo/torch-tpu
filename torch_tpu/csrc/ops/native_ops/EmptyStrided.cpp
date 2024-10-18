#include <torch/library.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUAllocator.h>

#include "TPUTorchUtils.h"
#include "common/config.h"
#include "TPUDeviceUtils.h"
#include "TPUStorageImpl.h"
#include "TPUTensorImpl.h"
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"

namespace at
{
Tensor empty_strided_tpu ( IntArrayRef                size,
                           IntArrayRef                stride,
                           c10::optional<ScalarType>  dtype_opt,
                           c10::optional<Layout>      layout_opt,
                           c10::optional<Device>      device_opt,
                           c10::optional<bool>        pin_memory_opt )
{
  TIMING_START;
  torch_tpu::utils::maybe_initialize_tpu(device_opt);
  auto scalar_type = dtype_or_default ( dtype_opt );
  // auto pin_memory = pinned_memory_or_default ( pin_memory_opt );
  at::detail::check_size_nonnegative ( size );
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta ( scalar_type );
  auto size_bytes = at::detail::computeStorageNbytes ( size, stride, dtype.itemsize() );
  auto allocator = c10::GetTPUAllocator();
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_tpu::TPUStorageImpl> (
                      c10::StorageImpl::use_byte_size_t(),
                      size_bytes,
                      allocator->allocate ( size_bytes ),
                      allocator,
                      /*resizeable=*/true );
  auto tensor = detail::make_tensor<torch_tpu::TPUTensorImpl> ( storage_impl, dtype );
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides ( size, stride );
  at_tpu::StorageDescHelper::SetDesc(tensor, size, tensor.strides());
  TIMING_END(tpu::MALLOC);
  SHOW_EMPTY_INFO(tensor);
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
  TIMING_START;
  torch_tpu::utils::maybe_initialize_tpu(device_opt);
  auto scalar_type = dtype_or_default ( dtype_opt );
  // auto pin_memory = pinned_memory_or_default ( pin_memory_opt );
  constexpr c10::DispatchKeySet ks ( c10::DispatchKey::TPU );
  auto allocator = c10::GetTPUAllocator();
  at::detail::check_size_nonnegative ( size );
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta ( scalar_type );
  size_t size_bytes = at::detail::computeStorageNbytesContiguous ( size, dtype.itemsize() );
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_tpu::TPUStorageImpl> (
                      c10::StorageImpl::use_byte_size_t(),
                      size_bytes,
                      allocator->allocate ( size_bytes ),
                      allocator,
                      /*resizeable=*/true );
  auto tensor = detail::make_tensor<torch_tpu::TPUTensorImpl> ( storage_impl, dtype );
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
      tensor.unsafeGetTensorImpl()->empty_tensor_restride ( *memory_format_opt );
    }
  }
  at_tpu::StorageDescHelper::SetDesc(tensor, size, tensor.strides());
  TIMING_END(tpu::MALLOC);
  SHOW_EMPTY_INFO(tensor);
  return tensor;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "empty.memory_format", empty_memory_format_tpu );
}
} // namespace at
