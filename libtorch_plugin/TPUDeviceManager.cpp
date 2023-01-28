#include <vector>
#include <unordered_map>
#include <mutex>
#include <c10/util/Logging.h>
#include <TPUDeviceManager.h>

#define ERROR_CODE(err) " ( TPU error code: " << err << ")"

namespace c10
{
namespace tpu
{

class TPUDeviceManager
{
public:
  TPUDeviceManager()
  {
    bm_status_t Status = BM_SUCCESS;
    int DeviceCount = 0;
    Status = bm_dev_getcount ( &DeviceCount );
    if ( Status != BM_SUCCESS )
    {
      LOG ( FATAL ) << "Failed to get TPU device count"
                    << ERROR_CODE ( Status );
    }
    if ( DeviceCount > 0 )
    {
      Handles_.resize ( DeviceCount, nullptr );
      for ( int i = 0; i < DeviceCount; ++i )
      {
        Status = bm_dev_request ( &Handles_[i], i );
        if ( Status != BM_SUCCESS )
        {
          LOG ( FATAL ) << "Failed to request tpu device #" << i
                        << ERROR_CODE ( Status );
        }
      }
      Index_ = 0;
    }
    else
    {
      Index_ = -1;
    }
    static std::mutex Mutex;
    Mutex.lock();
    if ( MemMap_.empty() && DeviceCount > 0 )
    {
      MemMap_.resize ( DeviceCount );
    }
    Mutex.unlock();
  }

  ~TPUDeviceManager()
  {
    for ( auto it : Handles_ )
    {
      bm_dev_free ( it );
    }
  }

  int GetDeviceIndex() const
  {
    return Index_;
  }

  void SetDeviceIndex ( int Index )
  {
    Index_ = Index;
  }

  int GetDeviceCount() const
  {
    return ( int ) Handles_.size();
  }

  bm_handle_t GetDeviceHandle() const
  {
    if ( Index_ < 0 || Index_ > ( int ) Handles_.size() - 1 )
    {
      return nullptr;
    }
    else
    {
      return Handles_[Index_];
    }
  }

  void * Alloc ( size_t Size ) const
  {
    Mutex_.lock();
    if ( Size == 0 ) { return nullptr; }
    if ( Size >= ( 1UL << 32 ) )
    {
      LOG ( FATAL ) << "TPU only allows to allocate memory with size "
                    << "smaller than (2^32) bytes";
    }
    bm_handle_t Handle = GetDeviceHandle();
    if ( Handle == nullptr )
    {
      LOG ( FATAL ) << "TPU handle of device #" << GetDeviceIndex()
                    << " is null";
    }
    bm_status_t Status = BM_SUCCESS;
    bm_device_mem_t Mem;
    Status = bm_malloc_device_byte ( Handle, &Mem, ( unsigned int ) Size );
    if ( Status != BM_SUCCESS )
    {
      LOG ( FATAL ) << "Failed to allocate memory on TPU device #"
                    << GetDeviceIndex() << " with size = " << Size << "bytes"
                    << ERROR_CODE ( Status );
    }
    MemoryInfo Info = { Handle, Mem };
    MemMap_[GetDeviceIndex()].emplace ( Mem.u.device.device_addr, Info );
    Mutex_.unlock();
    return ( void * ) Mem.u.device.device_addr;
  }

  void Free ( void * Ptr ) const
  {
    Mutex_.lock();
    if ( Ptr == nullptr )
    {
      return;
    }
    bm_handle_t Handle = GetDeviceHandle();
    if ( Handle == nullptr )
    {
      LOG ( FATAL ) << "TPU handle of device #" << GetDeviceIndex()
                    << " is null";
    }
    int Index = bm_get_devid ( Handle );
    auto Info = MemMap_[Index].find ( ( unsigned long long ) Ptr );
    if ( Info == MemMap_[Index].end() )
    {
      LOG ( FATAL ) << "Memory is not allocated on TPU device #"
                    << Index << " with address = " << Ptr;
    }
    bm_free_device ( Info->second.Handle, Info->second.Mem );
    MemMap_[Index].erase ( Info );
    Mutex_.unlock();
  }

  void CopyHostToDevice ( void * Dst, const void * Src, size_t Size )
  {
    bm_handle_t Handle = GetDeviceHandle();
    if ( Handle == nullptr )
    {
      LOG ( FATAL ) << "TPU handle of device #" << GetDeviceIndex()
                    << " is null";
    }
    bm_device_mem_t DstMem = bm_mem_from_device (
                             ( unsigned long long ) Dst, Size );
    bm_status_t Status = BM_SUCCESS;
    Status = bm_memcpy_s2d ( Handle, DstMem, ( void * ) Src );
    if ( Status != BM_SUCCESS )
    {
      LOG ( FATAL ) << "Failed to copy memory from host to TPU device #"
                    << GetDeviceIndex() << " with size = " << Size << "bytes"
                    << ERROR_CODE ( Status );
    }
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size )
  {
    bm_handle_t Handle = GetDeviceHandle();
    if ( Handle == nullptr )
    {
      LOG ( FATAL ) << "TPU handle of device #" << GetDeviceIndex()
                    << " is null";
    }
    bm_device_mem_t SrcMem = bm_mem_from_device (
                             ( unsigned long long ) Src, Size );
    bm_status_t Status = BM_SUCCESS;
    Status = bm_memcpy_d2s ( Handle, Dst, SrcMem );
    if ( Status != BM_SUCCESS )
    {
      LOG ( FATAL ) << "Failed to copy memory from TPU device #"
                    << GetDeviceIndex() << " to host with size = " << Size
                    << "bytes" << ERROR_CODE ( Status );
    }
  }

private:
  std::vector<bm_handle_t> Handles_;
  int Index_;
  struct MemoryInfo
  {
    bm_handle_t      Handle;
    bm_device_mem_t  Mem;
  };
  static std::vector < std::unordered_map < unsigned long long, MemoryInfo > >
  MemMap_;
  static std::mutex Mutex_;
};

std::vector < std::unordered_map < unsigned long long,
    TPUDeviceManager::MemoryInfo > > TPUDeviceManager::MemMap_;

std::mutex TPUDeviceManager::Mutex_;

static thread_local TPUDeviceManager ThreadLocalTPUDeviceManager;

int TPUGetDeviceCount ( void )
{
  return ThreadLocalTPUDeviceManager.GetDeviceCount();
}

int TPUGetDeviceIndex ( void )
{
  return ThreadLocalTPUDeviceManager.GetDeviceIndex();
}

void TPUSetDeviceIndex ( int Index )
{
  ThreadLocalTPUDeviceManager.SetDeviceIndex ( Index );
}

bm_handle_t TPUGetDeviceHandle()
{
  return ThreadLocalTPUDeviceManager.GetDeviceHandle();
}

void * TPUAlloc ( size_t Size )
{
  return ThreadLocalTPUDeviceManager.Alloc ( Size );
}

void TPUFree ( void * Ptr )
{
  ThreadLocalTPUDeviceManager.Free ( Ptr );
}

void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size )
{
  ThreadLocalTPUDeviceManager.CopyHostToDevice ( Dst, Src, Size );
}

void TPUCopyDeviceToHost ( void *Dst, const void *Src, size_t Size )
{
  ThreadLocalTPUDeviceManager.CopyDeviceToHost ( Dst, Src, Size );
}

} // namespace tpu
} // namespace c10
