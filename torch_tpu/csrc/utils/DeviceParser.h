#pragma once

#include <c10/core/Device.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/Device.h>

#include "torch_tpu/csrc/utils/Device.h"
namespace at_tpu {
namespace key {

static constexpr c10::DeviceType NativeDeviceType = c10::DeviceType::PrivateUse1;
static constexpr c10::DispatchKey NativeDispatchKey = c10::DispatchKey::PrivateUse1;
static constexpr c10::DispatchKey NativeAutogradDispatchKey = c10::DispatchKey::AutogradPrivateUse1;
static constexpr c10::Backend NativeBackend = c10::Backend::PrivateUse1;
static const std::string tpu_device_str = "tpu";
static const std::string default_device_str = "privateuseone";

static bool isDeviceTensor(const at::Tensor& tensor) {
    return tensor.device().type() == NativeDeviceType;
}


static at::Device parse_tpu_device(PyObject* obj) {
    if (!obj || obj == Py_None){
        return at::Device(c10::backendToDeviceType(c10::dispatchKeyToBackend(torch::tensors::get_default_dispatch_key())));
    }
    if (THPUtils_checkLong(obj)) {
        const auto device_index = THPUtils_unpackLong(obj);
        TORCH_CHECK(device_index >= 0, "Device index must not be negative");
        return at::Device(at_tpu::key::NativeDeviceType, device_index);
    }
    if (THPUtils_checkString(obj)) {
        std::string device_str = THPUtils_unpackString(obj);
        if (device_str.find(at_tpu::key::tpu_device_str) != std::string::npos){
            device_str = device_str.replace(device_str.find(at_tpu::key::tpu_device_str),
                                            at_tpu::key::tpu_device_str.length(),
                                            at_tpu::key::default_device_str);
        }
        return at::Device(device_str);
    }
    if (THPDevice_Check(obj)){
        const auto device = reinterpret_cast<THPDevice*>(obj);
        return device->device;
    }
    const auto device = reinterpret_cast<THPDevice*>(obj);
    return device->device;
}

static c10::optional<at::Device>  parse_tpu_device_optional(PyObject* obj) {
  if (!obj) {
    return c10::nullopt;
  }
  return parse_tpu_device(obj);
}

static at::Device parse_tpu_device_with_default(PyObject* obj, const at::Device& default_device) {
    if (!obj) return default_device;
    return parse_tpu_device(obj);
}

}; // namespace key
}; // namespace at_tpu