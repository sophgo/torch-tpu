#pragma once

#include <c10/core/Device.h>
namespace torch_tpu {

// device init related funcs
void init_tpu(const c10::DeviceIndex device_index = 0);
void init_tpu(const std::string& device_str);
void init_tpu(const at::Device& device);

// device finalize related funcs
void finalize_tpu();

} // namespace torch_tpu


namespace torch {
namespace tpu {

// device synchronize
void synchronize(int64_t device_index = -1);

} // namespace tpu
} // namespace torch


namespace c10 {
namespace tpu {

DeviceIndex current_device();

} // namespace tpu
} // namespace c10
