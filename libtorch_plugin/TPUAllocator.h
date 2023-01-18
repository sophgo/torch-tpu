#pragma once

#include <mutex>

#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>

namespace c10
{
using MemoryDeleter = void ( * ) ( void * );

// A simple struct that is used to report C10's memory allocation,
// deallocation status and out-of-memory events to the profiler
class C10_API ProfiledTPUMemoryReporter
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

C10_API ProfiledTPUMemoryReporter & profiledTPUMemoryReporter();

// Get the TPU Allocator.
C10_API at::Allocator * GetTPUAllocator();

// Sets the TPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetTPUAllocator ( at::Allocator * alloc, uint8_t priority = 0 );

// Get the Default TPU Allocator
C10_API at::Allocator * GetDefaultTPUAllocator();
} // namepsace c10
