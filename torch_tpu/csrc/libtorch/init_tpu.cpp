#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"
#include "torch_tpu/csrc/core/TPUGuard.h"


namespace torch_tpu {

bool is_tpu_device(const at::Device& device) {
  return device.type() == c10::DeviceType::PrivateUse1;
}


void init_tpu(const c10::DeviceIndex device_index) {
  auto status = tpu::InitTPUMgr();
  if (status != tpu::INIT_SUCCESS) {
    SOPHON_LOGE("init device failed");
    return;
  }
}


void init_tpu(const std::string& device_str) {
  auto device = at::Device(device_str);
  TORCH_CHECK(is_tpu_device(device), "tpu device init fail, except got tpu device, but got ", device_str);
  init_tpu(device.index());
}


void init_tpu(const at::Device& device) {
  TORCH_CHECK(is_tpu_device(device), "tpu device init fail, except got tpu device, but got ", str(device));
  init_tpu(device.index());
}

void finalize_tpu() {
  if (tpu::IsTPUMgrInited() == tpu::INIT_ALREADY) {
#ifdef BACKEND_SG2260
    try {
      c10_tpu::tpuSynchronizeDevice();
    } catch (std::exception& e) {
      TORCH_CHECK(false, "tpu SynchronizeDevice failed err=:%s", e.what());
    }
#endif
    auto status = tpu::DestoryTpuMgr();
    if (status != tpu::DESTORY_SUCCESS) {
      TORCH_CHECK(false, "tpu finalize failed.\n");
    }
  } else {
    TORCH_WARN("Please init tpu device first!");
  }
}

} // namespace torch_tpu


namespace torch_tpu {

void synchronize(int64_t device_index) {
  c10_tpu::TPUGuard device_guard(at::Device(at::DeviceType::PrivateUse1, device_index));
#ifdef BACKEND_SG2260
  c10_tpu::tpuSynchronizeDevice();
#endif
}

} // namespace torch_tpu
