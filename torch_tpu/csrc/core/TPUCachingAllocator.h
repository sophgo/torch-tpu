#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <optional>

#include "TPUAllocator.h"


namespace c10_tpu {

namespace TPUCachingAllocator {

class TPUAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static bool release_lock_on_tpumalloc() {
    return instance().m_release_lock_on_tpumalloc;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_TPU_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static std::vector<size_t> roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions;
  }

  static size_t max_non_split_rounding_size() {
    return instance().m_max_non_split_rounding_size;
  }

  static TPUAllocatorConfig& instance() {
    static TPUAllocatorConfig* s_instance = ([]() {
      auto inst = new TPUAllocatorConfig();
      auto env = std::getenv("PYTORCH_TPU_ALLOC_CONF");
      std::optional<std::string> opt;
      if (env)
        opt = env;
      inst->parseArgs(opt);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const std::optional<std::string>& env);

 private:
  TPUAllocatorConfig();

  static void lexArgs(const std::string& env, std::vector<std::string>& config);
  static void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseMaxNonSplitRoundingSize(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);

  std::atomic<size_t> m_max_split_size;
  std::atomic<size_t> m_max_non_split_rounding_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<bool> m_release_lock_on_tpumalloc;
};

struct Block;
class DeviceCachingAllocator;

class CachingTPUAllocator : public TPUAllocator {
 private:
  mutable std::mutex mutex;
  mutable ska::flat_hash_map<void*, Block*> allocated_blocks;


  // Start caching right away
  // If not, all cache will be released before next allocation or after next free
  bool start_cache_ = true;

  void add_allocated_block(Block* block) const;

  Block* get_allocated_block(void* ptr, bool remove = false);

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;

  void init() override;

  void malloc(void** devPtr, c10::DeviceIndex device, size_t size) const;

  void free(void* ptr) override;

  void emptyCache(MempoolId_t mempool_id) override;

  void startCache(MempoolId_t mempool_id) override;
  void stopCache(MempoolId_t mempool_id) override;

  c10::DataPtr allocate(size_t size) const override;

  at::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t size) override;

  void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(tpuStream_t)> filter) override;

  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) override;

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override;

  void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    TPUAllocator* allocator_ptr = nullptr) override;

  void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) override;

  int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) override;
};

// General caching allocator utilities
void setAllocatorSettings(const std::string& env);

TPUAllocator* GetCachingTPUAllocator();

inline void emptyCache(MempoolId_t mempool_id = {0, 0}) {
  return GetCachingTPUAllocator()->emptyCache(mempool_id);
}

inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(tpuStream_t)> filter) {
  GetCachingTPUAllocator()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  GetCachingTPUAllocator()->endAllocateToPool(device, mempool_id);
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return GetCachingTPUAllocator()->releasePool(device, mempool_id);
}

inline void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    TPUAllocator* allocator_ptr = nullptr) {
  GetCachingTPUAllocator()->createOrIncrefPool(device, mempool_id, allocator_ptr);
}

inline void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) {
  GetCachingTPUAllocator()->setUseOnOOM(device, mempool_id);
}

inline int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return GetCachingTPUAllocator()->getPoolUseCount(device, mempool_id);
}

} // namespace c10_tpu::TPUCachingAllocator


// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the TPUCachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct MemPool {
  MemPool(
      TPUAllocator* allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  TPUAllocator* allocator();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  TPUAllocator* allocator_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
};

} // namespace c10_tpu
