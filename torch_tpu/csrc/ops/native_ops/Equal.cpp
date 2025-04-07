#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>

#include <ATen/NativeFunctions.h>

namespace at {

bool tpu_equal(const Tensor& self, const Tensor &src) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }

  // This is the same optimization done in the cpu_equal. Since the flags like neg/conj should be already handled outside the
  // cuda_equal, it should be safe to have the following fast path by
  // ensuring the storage and strides exactly the same.
  if (self.is_alias_of(src)
      && self.storage_offset() == src.storage_offset()
      && self.dtype() == src.dtype()
      && self.is_contiguous() == src.is_contiguous()
      && self.strides().equals(src.strides())
      // Extra checks to ensure the safety in case cuda_equal is directly called in C++.
      && self.layout() == src.layout()
      && self.is_neg() == src.is_neg()
      && self.is_conj() == src.is_conj()) {
    return true;
  }

  return eq(self, src).all().item().to<bool>();
}

// https://pytorch.org/docs/2.1/generated/torch.equal.html#torch.equal
TORCH_LIBRARY_IMPL(aten, TPU, m) {
    m.impl("equal", tpu_equal);
}

} // namespace at
