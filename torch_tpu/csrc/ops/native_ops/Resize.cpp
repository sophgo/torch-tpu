#include "Resize.h"
#include "TPUTorchUtils.h"

namespace at {
  const Tensor& resize_tpu_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return at::native::resize_named_tensor_(self, size, optional_memory_format);
  }
  at::OptionalIntArrayRef stride_opt = c10::nullopt;
  at::native::resize_impl_tpu_(self.unsafeGetTensorImpl(), size, stride_opt, /*resize_storage=*/true);
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    if (memory_format == MemoryFormat::Preserve) {
      memory_format = self.suggest_memory_format();
    }
    self.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }
  return self;
  }

  TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("resize_", resize_tpu_);
  }
} // namespace at
