#include "torch_tpu/csrc/core/TPUFunction.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"

using namespace c10;

namespace c10_tpu{

DeviceIndex device_count() noexcept
{
    int count = tpu::TPUGetDeviceCount();
    return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device()
{
    int idx = tpu::TPUGetDeviceIndex();
    return static_cast<DeviceIndex>(idx);
}

void set_device(DeviceIndex dev_id)
{
    tpu::TPUSetDeviceIndex(dev_id);
}


}; // namespace c10_tpu
