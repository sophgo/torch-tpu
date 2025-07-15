#include "KernelManager.hpp"
#include "torch_tpu_kernel_data.h"

namespace tpu {
std::shared_ptr<KernelManager> KernelManager::instance_ = nullptr;
KernelManager::KernelManager() {
  kernel_file_ = getenv("TPUKERNEL_FIRMWARE_PATH");
  if (kernel_file_) {
    LOG(INFO) << "Using kernel file: " << kernel_file_;
  }
}

KernelManager::~KernelManager() {
  for (auto &it : module_map_) {
    tpuKernelUnloadModule(it.second, it.first);
  }
}

tpuKernelModule_t KernelManager::RegisterKernelModule(tpuStream_t stream) {
  if (module_map_.find(stream) != module_map_.end()) {
    return module_map_[stream];
  }
  tpuKernelModule_t module;
  if (kernel_file_) {
    module = tpuKernelModuleLoadFromFile(kernel_file_, stream);
    if (module == nullptr) {
      LOG(ERROR) << "Failed to load kernel module from " << kernel_file_;
      return nullptr;
    }
  } else {
    module = tpuKernelModuleLoad(torch_tpu_kernel_data,
                                 torch_tpu_kernel_data_length, stream);
    if (module == nullptr) {
      LOG(ERROR) << "Failed to load kernel module from torch_tpu_kernel_data";
      return nullptr;
    }
  }
  module_map_[stream] = module;
  if (getenv("TorchTpuSaveKernelModule") != nullptr) {
    auto path = KernelModuleSavePath();
    SaveKernelModule(path, module);
  }
  return module;
}

tpuError_t KernelManager::UnloadKernelModule(tpuStream_t stream) {
  auto ret = tpuSuccess;
  if (module_map_.find(stream) != module_map_.end()) {
    ret = tpuKernelUnloadModule(module_map_[stream], stream);
    module_map_.erase(stream);
  }
  return ret;
}
} // namespace tpu