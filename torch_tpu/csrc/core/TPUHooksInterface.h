#pragma once
#include <c10/core/Storage.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include "torch_tpu/csrc/aten/TPUGeneratorImpl.h"

namespace c10_tpu {

struct TORCH_API TPUHooksInterface : public at::PrivateUse1HooksInterface {
    virtual ~TPUHooksInterface() = default;
    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
      static auto device_gen = at_tpu::detail::getDefaultTPUGenerator(device_index);
      return device_gen;
    };
    bool isAvailable() const override { 
      return tpu::IsTPUMgrInited() == tpu::INIT_ALREADY;
    };

    bool hasPrimaryContext(DeviceIndex device_index) const override {
      int current_device_idx = tpu::TPUGetDeviceIndex();
      return device_index == current_device_idx;
    }
    void init() const override;
};

struct TORCH_API TPUHooksArgs : public at::PrivateUse1HooksArgs {};
at::PrivateUse1HooksInterface* get_tpu_hooks();
}
