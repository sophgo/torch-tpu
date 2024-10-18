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

inline void torch_check_npu(const at::Tensor& tensor) {
  TORCH_CHECK(is_tpu(tensor),
              "Expected TPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::TensorOptions& options) {
  TORCH_CHECK(is_tpu(options),
              "Expected TPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::Device& device) {
  TORCH_CHECK(is_tpu(device),
              "Expected TPU tensor, please check whether the input tensor device is correct.");
}

inline void maybe_initialize_tpu(at::TensorOptions& options) {
    if (torch_tpu::utils::is_tpu(options)) {
        if (options.device().index() < 0) {
          options.device().set_index(0);
        }
        auto status = tpu::TPUDeviceInitialize(options.device().index());
        if (status != tpu::INIT_SUCCESS) {
        TORCH_CHECK(false, "tpu device ", options.device().index(), " init failed.");
        }
        #ifndef BUILD_LIBTORCH
        torch_tpu::utils::tpu_lazy_init();
        #endif
    }
}

inline void maybe_initialize_tpu(at::Device& device) {
  if (torch_tpu::utils::is_tpu(device)) {
    if (device.index() < 0) {
      device.set_index(0);
    }
        auto status = tpu::TPUDeviceInitialize(device.index());
    if (status != tpu::INIT_SUCCESS) {
      TORCH_CHECK(false, "tpu device ", device.index(), " init failed.");
    }
#ifndef BUILD_LIBTORCH
    torch_tpu::utils::tpu_lazy_init();
#endif
  }
}

inline void maybe_initialize_tpu(c10::optional<at::Device>& device) {
  if (device) {
    maybe_initialize_tpu(*device);
  }
}
} // namespace utils
} // namespace torch_tpu

