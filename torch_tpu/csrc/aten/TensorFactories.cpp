#include "TPUNativeFunctions.h"
#include "torch_tpu/csrc/core/TPUDeviceUtils.h"
#include "torch_tpu/csrc/core/TPUAllocator.h"
#include "torch_tpu/csrc/core/TPUStorageImpl.h"
#include "torch_tpu/csrc/core/TPUTensorImpl.h"
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"

namespace at_tpu {
namespace TPUNativeFunctions{
    at::Tensor empty_with_format(c10::IntArrayRef size,
                                c10::optional<at::ScalarType> dtype_opt,
                                c10::optional<c10::Layout> layout_opt,
                                c10::optional<c10::Device> device_opt,
                                c10::optional<bool> pin_memory_opt,
                                int64_t dst_format) {
        auto device_ = c10::device_or_default(device_opt);
        torch_tpu::utils::is_tpu(device_);
        torch_tpu::utils::maybe_initialize_tpu(device_);
        TORCH_CHECK(!pinned_memory_or_default(pin_memory_opt), "Only dense CPU tensors can be pinned");

        auto allocator = c10::GetTPUAllocator();
        int64_t nelements = at_tpu::StorageDescHelper::GetMemorySize(size, (torch_tpu::tpuFormat)dst_format);
        auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
        int64_t size_bytes = nelements * dtype.itemsize();
        c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_tpu::TPUStorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            size_bytes,
            allocator->allocate(size_bytes),
            allocator,
            true);

        auto tensor =
            at::detail::make_tensor<torch_tpu::TPUTensorImpl>(storage_impl, dtype);
        // Default TensorImpl has size [0]
        if ( size.size() != 1 || size[0] != 0 )
        {
            tensor.unsafeGetTensorImpl()->set_sizes_contiguous ( size ); // Only support Contiguous Layout
        }
        at_tpu::StorageDescHelper::SetDesc(tensor, size, tensor.strides(), static_cast<torch_tpu::tpuFormat>(dst_format));
        return tensor;
    }

    at::Tensor unsafe_empty_with_format(c10::IntArrayRef size,
                                    c10::optional <at::ScalarType> dtype_opt,
                                    c10::optional <c10::Layout> layout_opt,
                                    c10::optional <c10::Device> device_opt,
                                    c10::optional<bool> pin_memory_opt,
                                    int64_t dst_format,
                                    bool keep_format) {
        // This is a special interface that can adjust the memory application results. Check before use.

        // Some ops cannot operate directly based on ND format, such as MatMul, BatchMatMul.
        // For these ops, specify the parameter keep_format to ensure that
        // the specified internal format is preserved.
        // keep_format attribute  ????????????
        return empty_with_format(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, dst_format);
    }

    static void DummyReportAndDelete(void *ptr){}
    at::Tensor make_tensor_from_ptr(void *ptr, std::vector<int64_t>& sizes, at::ScalarType dtype)
    {
        auto tensor_dtype = at::scalarTypeToTypeMeta(dtype);
        at::DataPtr ptr_ = {ptr, ptr, &DummyReportAndDelete, tpu::TPUGetCurrentDevice()};
        auto allocator = c10::GetTPUAllocator();
        c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_tpu::TPUStorageImpl> (
                            c10::StorageImpl::use_byte_size_t(),
                            0,
                            std::move(ptr_),
                            allocator,
                            /*resizeable=*/false );
        auto tensor = at::detail::make_tensor<torch_tpu::TPUTensorImpl> ( storage_impl, tensor_dtype );
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous ( sizes );
        return tensor;
    }


} // namespace TPUNativeFunctions
} // namespace at_tpu