#pragma once

#include <mutex>

#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include "TPUStream.h"
#include "tpu_runtime_api.h"
#include <cstring>


namespace c10_tpu
{
using MemoryDeleter = void ( * ) ( void * );

using CaptureId_t = unsigned long long;

// first is set if the instance is created by TPUGraph::capture_begin.
// second is set if the instance is created by graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

class TPUAllocator : public c10::Allocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void emptyCache(MempoolId_t mempool_id = {0, 0}) = 0;
  virtual void startCache(MempoolId_t mempool_id = {0, 0}) = 0;
  virtual void stopCache(MempoolId_t mempool_id = {0, 0}) = 0;
  virtual void init() {
    printf("NOT need initlization\n");
    return;
  };
  virtual void free(void* ptr) {
    TORCH_CHECK(
      false,
      " does not yet support free. ");
  };
  virtual void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
    TORCH_CHECK(
      false,
      " does not yet support cacheInfo. ");
  };
  virtual void recordStream(const c10::DataPtr&, TPUStream stream) {
    TORCH_CHECK(
      false,
      " does not yet support recordStream. ");
  };
  virtual void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(tpuStream_t)> filter) {
    TORCH_CHECK(
      false,
      " does not yet support beginAllocateToPool. ");
  };
  virtual void endAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id) {
    TORCH_CHECK(
      false,
      " does not yet support endAllocateToPool. ");
  };
  virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
      false,
      " does not yet support releasePool. ");
  };
  virtual int getPoolUseCount(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/) {
    TORCH_CHECK(
        false,
        " does not yet support getPoolUseCount. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void createOrIncrefPool(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      TPUAllocator* allocator = nullptr) {
    TORCH_CHECK(
        false,
        " does not yet support createOrIncrefPool. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
        false,
        " does not yet support setUseOnOOM. "
        "If you need it, please file an issue describing your use case.");
  }

};

// A simple struct that is used to report C10's memory allocation,
// deallocation status and out-of-memory events to the profiler
class ProfiledTPUMemoryReporter
{
public:
  ProfiledTPUMemoryReporter() {}
  void New ( void * ptr, size_t nbytes );
  void OutOfMemory ( size_t nbytes );
  void Delete ( void * ptr );

private:
  std::mutex mutex_;
  std::unordered_map<void *, size_t> size_table_;
  size_t allocated_ = 0;
  size_t log_cnt_ = 0;
};

ProfiledTPUMemoryReporter & profiledTPUMemoryReporter();

// Get the TPU Allocator.
at::Allocator * GetTPUAllocator();

// Get the Default TPU Allocator
TPUAllocator* GetDefaultTPUAllocator();

void EmptyCache();
void StartCache();
void StopCache();
} // namepsace c10_tpu
