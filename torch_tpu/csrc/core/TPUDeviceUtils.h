#pragma once
#include <c10/core/TensorOptions.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include "torch_tpu/csrc/core/TPUDeviceManager.h"
#ifndef BUILD_LIBTORCH
#include "torch_tpu/csrc/utils/LazyInit.h"
#endif

namespace torch_tpu {
namespace utils {
inline bool is_tpu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}

inline bool is_tpu(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}

inline bool is_tpu(const at::Device& device) {
  return device.is_privateuseone();
}

inline void maybe_initialize_tpu(const at::TensorOptions& options) {
    if (torch_tpu::utils::is_tpu(options)) {
        auto status = tpu::TPUDeviceInitialize(options.device().index());
        if (status != tpu::INIT_SUCCESS) {
        TORCH_CHECK(false, "tpu device ", options.device().index(), " init failed.");
        }
        #ifndef BUILD_LIBTORCH
        torch_tpu::utils::tpu_lazy_init();
        #endif
    }
}

inline void maybe_initialize_tpu(const at::Device& device) {
  if (torch_tpu::utils::is_tpu(device)) {
        auto status = tpu::TPUDeviceInitialize(device.index());
    if (status != tpu::INIT_SUCCESS) {
      TORCH_CHECK(false, "tpu device ", device.index(), " init failed.");
    }
#ifndef BUILD_LIBTORCH
    torch_tpu::utils::tpu_lazy_init();
#endif
  }
}

inline void maybe_initialize_tpu(const c10::optional<at::Device>& device) {
  if (device) {
    maybe_initialize_tpu(*device);
  }
}
} // namespace utils 
} // namespace torch_tpu

