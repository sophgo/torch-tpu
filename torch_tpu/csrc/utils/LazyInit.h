#pragma once
#include <c10/core/TensorOptions.h>

#include "torch_tpu/csrc/utils/DeviceParser.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"

namespace torch_tpu {
namespace utils {

void tpu_lazy_init();

void tpu_set_run_yet_variable_to_false();

static bool isTPUDevice(const at::TensorOptions& options) {
    return options.device().type() == at_tpu::key::NativeDeviceType;
};

static bool isTPUDevice(const at::Device& device) {
    return device.type() == at_tpu::key::NativeDeviceType;
}

static void maybe_initialize_tpu(const at::TensorOptions& options) {
    if (isTPUDevice(options)) {
      pybind11::gil_scoped_release no_gil;
      tpu::TPUSetDeviceIndex(options.device().index());
    }
    torch_tpu::utils::tpu_lazy_init();
}

static void maybe_initialize_tpu(const at::Device& device) {
    if (isTPUDevice(device)) {
      pybind11::gil_scoped_release no_gil;
      tpu::TPUSetDeviceIndex(device.index());   
    }
    torch_tpu::utils::tpu_lazy_init();
}

static void maybe_initialize_tpu(const c10::optional<at::Device>& device) {
  if (device) {
    maybe_initialize_tpu(*device);
  }
}
}; // namespace utils
}; // namespace torch_tpu