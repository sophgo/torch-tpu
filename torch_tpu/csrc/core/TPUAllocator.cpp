#include "TPUAllocator.h"
#include "TPUCachingAllocator.h"
#include "TPUTorchUtils.h"

namespace c10_tpu
{
struct DefaultTPUAllocator final : TPUAllocator
{
  DefaultTPUAllocator() = default;

  c10::DataPtr allocate ( size_t nbytes )  override
  {
    void * data = nullptr;
    try
    {
      data = tpu::TPUAlloc ( nbytes );
    }
    catch ( c10::Error & e )
    {
      profiledTPUMemoryReporter().OutOfMemory ( nbytes );
      throw e;
    }
    profiledTPUMemoryReporter().New ( data, nbytes );
    return { data, data, &ReportAndDelete, tpu::TPUGetCurrentDevice() };
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

  static void ReportAndDelete ( void * ptr )
  {
    if ( !ptr )
    {
      return;
    }
    profiledTPUMemoryReporter().Delete ( ptr );
    tpu::TPUFree ( ptr );
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void* raw_alloc(size_t nbytes) {
    return tpu::TPUAlloc(nbytes);
  }

  void emptyCache(MempoolId_t mempool_id = {0, 0}) {
    tpu::TPUEmptyCache( );
  }
  
  void startCache(MempoolId_t mempool_id = {0, 0}) {
    tpu::TPUEnableCache( );
  }

  void stopCache(MempoolId_t mempool_id = {0, 0}) {
    tpu::TPUDisableCache( );
  }

  private:
    static void raw_delete(void* ptr) {
      std::free(ptr);
    }
};

ProfiledTPUMemoryReporter & profiledTPUMemoryReporter()
{
  static ProfiledTPUMemoryReporter reporter_;
  return reporter_;
}

at::Allocator * GetTPUAllocator()
{
  return GetAllocator ( DeviceType::TPU );
}

// Global default TPU Allocator
static DefaultTPUAllocator g_tpu_alloc;

TPUAllocator * GetDefaultTPUAllocator()
{
  return &g_tpu_alloc;
}

void ProfiledTPUMemoryReporter::New ( void * ptr, size_t nbytes )
{
  if ( nbytes == 0 )
  {
    return;
  }
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if ( profile_memory )
  {
    std::lock_guard<std::mutex> guard ( mutex_ );
    size_table_[ptr] = nbytes;
    allocated_ += nbytes;
    allocated = allocated_;
  }
  if ( profile_memory )
  {
    auto Device = at::Device ( at::DeviceType::TPU );
    Device.set_index ( tpu::TPUGetDeviceIndex() );
    reportMemoryUsageToProfiler ( ptr, nbytes, allocated, 0, Device );
  }
}

void ProfiledTPUMemoryReporter::Delete ( void * ptr )
{
  size_t nbytes = 0;
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if ( profile_memory )
  {
    std::lock_guard<std::mutex> guard ( mutex_ );
    auto it = size_table_.find ( ptr );
    if ( it != size_table_.end() )
    {
      allocated_ -= it->second;
      allocated = allocated_;
      nbytes = it->second;
      size_table_.erase ( it );
    }
    else
    {
      // C10_LOG_EVERY_MS might log every time in some builds,
      // using a simple counter to avoid spammy logs
      if ( log_cnt_++ % 1000 == 0 )
      {
        LOG ( WARNING ) << "Memory block of unknown size was allocated before "
                        << "the profiling started, profiler results will not "
                        << "include the deallocation event";
      }
    }
  }
  if ( nbytes == 0 )
  {
    return;
  }
  if ( profile_memory )
  {
    auto Device = at::Device ( at::DeviceType::TPU );
    Device.set_index ( tpu::TPUGetDeviceIndex() );
    reportMemoryUsageToProfiler ( ptr, -nbytes, allocated, 0, Device );
  }
}

void ProfiledTPUMemoryReporter::OutOfMemory ( size_t nbytes )
{
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if ( profile_memory )
  {
    std::lock_guard<std::mutex> guard ( mutex_ );
    allocated = allocated_;
  }
  if ( nbytes == 0 )
  {
    return;
  }
  if ( profile_memory )
  {
    auto Device = at::Device ( at::DeviceType::TPU );
    Device.set_index ( tpu::TPUGetDeviceIndex() );
    reportOutOfMemoryToProfiler (
    static_cast<int64_t> ( nbytes ),
    static_cast<int64_t> ( allocated ),
    0,
    Device );
  }
}

class DynamicAllocatorProxy : public TPUAllocator {
 public:
  DynamicAllocatorProxy() {
    const char* env_val = std::getenv("PYTORCH_TPU_ALLOCATOR");
    use_caching_ = (env_val && std::string(env_val) == "caching");
  }

  at::DataPtr allocate(size_t size) override {
    ensureInit();
    return target()->allocate(size);
  }

  at::DeleterFnPtr raw_deleter() const override {
    ensureInit();
    return target()->raw_deleter();
  }

  void* raw_alloc(size_t nbytes) {
    ensureInit();
    return target()->raw_alloc(nbytes);
  }

  void emptyCache(MempoolId_t mempool_id = {0, 0}) {
    ensureInit();
    target()->emptyCache();
  }

  void startCache(MempoolId_t mempool_id = {0, 0}) {
    ensureInit();
    target()->startCache();
  }

  void stopCache(MempoolId_t mempool_id = {0, 0}) {
    ensureInit();
    target()->stopCache();
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

 private:

  void ensureInit() const {
    if (use_caching_) {
      std::call_once(init_flag_, [this]{ target()->init(); });
    }
  }
  TPUAllocator* target() const {
    return use_caching_ ? c10_tpu::TPUCachingAllocator::GetCachingTPUAllocator() : GetDefaultTPUAllocator();
  }
  mutable std::once_flag init_flag_;
  bool use_caching_ = false;
};

static DynamicAllocatorProxy g_dynamic_tpu_alloc;
REGISTER_ALLOCATOR(DeviceType::TPU, &g_dynamic_tpu_alloc);

void EmptyCache() {
  auto allocator = static_cast<DynamicAllocatorProxy*>(GetTPUAllocator());
  allocator->emptyCache();
}

void StartCache() {
  auto allocator = static_cast<DynamicAllocatorProxy*>(GetTPUAllocator());
  allocator->startCache();
}

void StopCache() {
  auto allocator = static_cast<DynamicAllocatorProxy*>(GetTPUAllocator());
  allocator->stopCache();
}

} // namespace c10_tpu
