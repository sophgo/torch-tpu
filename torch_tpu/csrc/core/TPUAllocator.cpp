#include <TPUAllocator.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>

namespace c10
{
struct C10_API DefaultTPUAllocator final : at::Allocator
{
  DefaultTPUAllocator() = default;

  at::DataPtr allocate ( size_t nbytes ) const override
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

at::Allocator * GetDefaultTPUAllocator()
{
  return &g_tpu_alloc;
}

REGISTER_ALLOCATOR ( DeviceType::TPU, &g_tpu_alloc );

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
} // namespace c10
