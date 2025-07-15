#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <tpu_runtime_api.h>
#include <unordered_map>

namespace tpu {

class KernelManager {
public:
  KernelManager();
  ~KernelManager();
  static std::shared_ptr<KernelManager> Instance() {
    static std::once_flag flag;
    std::call_once(flag,
                   [&]() { instance_ = std::make_shared<KernelManager>(); });
    return instance_;
  }

  tpuKernelModule_t RegisterKernelModule(tpuStream_t stream);
  tpuError_t UnloadKernelModule(tpuStream_t stream);

private:
  static std::shared_ptr<KernelManager> instance_;
  std::unordered_map<tpuStream_t, tpuKernelModule_t> module_map_;

  char *kernel_file_;

  void SaveKernelModule(const char *file, const tpuKernelModule_t &module) {
    FILE *handle = fopen(file, "wb");
    if (handle == nullptr) {
      LOG(ERROR) << "Failed to open file " << file;
    }
    size_t written = fwrite(module, 80, 1, handle);
    if (written != 1) {
      LOG(ERROR) << "Failed to write kernel module to file " << file;
    }
    fclose(handle);
  }
  const char *KernelModuleSavePath() {
    std::string path =
        std::string(getenv("HOME")) + "/.torch_tpu_kernel_module";
    return path.c_str();
  }
};

} // namespace tpu