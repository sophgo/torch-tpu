#pragma once

#include <c10/core/Device.h>

using namespace c10;

namespace c10_tpu{

DeviceIndex device_count() noexcept;

DeviceIndex current_device();

void set_device(DeviceIndex dev_id);

}; // namespace c10_tpu