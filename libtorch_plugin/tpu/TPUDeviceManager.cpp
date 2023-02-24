#include <vector>
#include <unordered_map>
#include <mutex>
#include <c10/util/Logging.h>
#include <TPUDeviceManager.h>

#include <iostream>

#define ERROR_CODE(Err) " ( TPU error code: " << Err << ")"
#define TPU_DEVICE_INDEX_BITS 6
#define TPU_GLOBAL_ADDR_BITS (64 - TPU_DEVICE_INDEX_BITS)

//#define SHOW_INFO

namespace tpu
{

static inline unsigned long long UnifiedAddr (
unsigned long long Addr, int Index )
{
  TORCH_CHECK ( Addr < ( 1UL << TPU_GLOBAL_ADDR_BITS ) );
  return ( ( ( unsigned long long ) Index ) << TPU_GLOBAL_ADDR_BITS ) | Addr;
}

static inline int GetDeviceIndexByUnifiedAddr ( unsigned long long Addr )
{
  return Addr >> TPU_GLOBAL_ADDR_BITS;
}

static inline unsigned long long GetAddrByUnifiedAddr ( unsigned long long Addr )
{
  return ( Addr << TPU_DEVICE_INDEX_BITS ) >> TPU_DEVICE_INDEX_BITS;
}

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

  bm_handle_t GetDeviceHandle ( int Index ) const
  {
    if ( Index < 0 || Index > ( int ) Handles_.size() - 1 )
    {
      return nullptr;
    }
    else
    {
      return Handles_[Index];
    }
  }

  bm_handle_t GetDeviceHandle() const
  {
    return GetDeviceHandle ( Index_ );
  }

  void * Alloc ( size_t Size ) const
  {
    Mutex_.lock();
    if ( Size == 0 )
    {
      Mutex_.unlock();
      return nullptr;
    }
#if 0
    auto Size_ = ( ( ( Size - 1 ) >> 16 ) + 1 ) << 16;
#else
    auto Size_ = Size;
#endif
    TORCH_CHECK ( Size_ < ( 1UL << 32 ), "TPU only allows to allocate memory with size smaller than (2^32) bytes" );
    bm_handle_t Handle = GetDeviceHandle();
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    bm_device_mem_t Mem;
    bm_status_t Status = bm_malloc_device_byte ( Handle, &Mem, ( unsigned int ) Size_ );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to allocate memory on TPU device #", GetDeviceIndex(), " size = ", Size_, "bytes" );
    unsigned long long Addr = UnifiedAddr ( Mem.u.device.device_addr, GetDeviceIndex() );
    AddrMemMap_.emplace ( Addr, Mem );
#ifdef SHOW_INFO
    std::cout << "Alloc addr = " << ( void * ) Addr << " size = " << Size << std::endl;
#endif
    Mutex_.unlock();
    return ( void * ) Addr;
  }

  void Free ( void * Ptr ) const
  {
    Mutex_.lock();
    if ( Ptr == nullptr )
    {
      Mutex_.unlock();
      return;
    }
    int Index = GetDeviceIndexByUnifiedAddr ( ( unsigned long long ) Ptr );
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    auto Iter = AddrMemMap_.find ( ( unsigned long long ) Ptr );
    TORCH_CHECK ( Iter != AddrMemMap_.end(), "Memory of address = ", Ptr, " is not found" );
    TORCH_CHECK ( GetAddrByUnifiedAddr ( ( unsigned long long ) Ptr ) == Iter->second.u.device.device_addr );
    bm_free_device ( Handle, Iter->second );
    AddrMemMap_.erase ( Iter );
#ifdef SHOW_INFO
    std::cout << "Free addr = " << Ptr << std::endl;
#endif
    Mutex_.unlock();
  }

  void CopyHostToDevice ( void * Dst, const void * Src, size_t Size )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Host = " << Src << " to Device = " << Dst << " Size = " << Size << std::endl;
#endif
    int Index = GetDeviceIndexByUnifiedAddr ( ( unsigned long long ) Dst );
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    bm_device_mem_t DstMem = bm_mem_from_device ( GetAddrByUnifiedAddr ( ( unsigned long long ) Dst ), Size );
    bm_status_t Status = bm_memcpy_s2d ( Handle, DstMem, ( void * ) Src );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from host to TPU device #", Index, " size = ", Size, "bytes" );
#if 0
    bm_device_sync ( Handle );
#endif
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Device = " << Src << " to Host = " << Dst << " Size = " << Size << std::endl;
#endif
    int Index = GetDeviceIndexByUnifiedAddr ( ( unsigned long long ) Src );
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    bm_device_mem_t SrcMem = bm_mem_from_device ( GetAddrByUnifiedAddr ( ( unsigned long long ) Src ), Size );
    bm_status_t Status = BM_SUCCESS;
    Status = bm_memcpy_d2s ( Handle, Dst, SrcMem );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from TPU device #", Index, " to host size = ", Size, "bytes" );
#if 0
    bm_device_sync ( Handle );
#endif
  }

  void CopyDeviceToDevice ( void * Dst, const void * Src, size_t Size )
  {
    int DstIndex = GetDeviceIndexByUnifiedAddr ( ( unsigned long long ) Dst );
    int SrcIndex = GetDeviceIndexByUnifiedAddr ( ( unsigned long long ) Src );
    bm_handle_t DstHandle = GetDeviceHandle ( DstIndex );
    if ( DstHandle == nullptr )
    {
      LOG ( FATAL ) << "TPU handle of device #" << DstIndex << " is null";
    }
    bm_handle_t SrcHandle = GetDeviceHandle ( SrcIndex );
    if ( SrcHandle == nullptr )
    {
      LOG ( FATAL ) << "TPU handle of device #" << SrcIndex << " is null";
    }
    bm_device_mem_t DstMem =
    bm_mem_from_device ( GetAddrByUnifiedAddr ( ( unsigned long long ) Dst ),
                         Size );
    bm_device_mem_t SrcMem =
    bm_mem_from_device ( GetAddrByUnifiedAddr ( ( unsigned long long ) Src ),
                         Size );
    bm_status_t Status = BM_SUCCESS;
    if ( DstIndex == SrcIndex )
    {
      Status = bm_memcpy_d2d_byte ( DstHandle, DstMem, 0, SrcMem, 0, Size );
    }
    else
    {
      Status = bm_memcpy_c2c ( SrcHandle, DstHandle, SrcMem, DstMem, false );
    }
    if ( Status != BM_SUCCESS )
    {
      LOG ( FATAL ) << "Failed to copy memory from TPU device #"
                    << SrcIndex << " to TPU device #" << DstIndex
                    << " with size = " << Size
                    << "bytes" << ERROR_CODE ( Status );
    }
#if 0
    if ( DstIndex == SrcIndex )
    {
      bm_device_sync ( DstHandle );
    }
    else
    {
      bm_device_sync ( DstHandle );
      bm_device_sync ( SrcHandle );
    }
#endif
  }

private:
  std::vector<bm_handle_t> Handles_;
  int Index_;
  static std::mutex Mutex_;
  static std::unordered_map<unsigned long long, bm_device_mem_t> AddrMemMap_;

};

std::mutex TPUDeviceManager::Mutex_;
std::unordered_map<unsigned long long, bm_device_mem_t> TPUDeviceManager::AddrMemMap_;

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

void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size )
{
  ThreadLocalTPUDeviceManager.CopyDeviceToHost ( Dst, Src, Size );
}

bool TPUPtrIsInCurrentDevice ( const void * Ptr )
{
  int Index = GetDeviceIndexByUnifiedAddr ( ( unsigned long long ) Ptr );
  return ThreadLocalTPUDeviceManager.GetDeviceIndex() == Index;
}

void * TPUGetAddrInDevice ( const void * Ptr )
{
  return ( void * ) GetAddrByUnifiedAddr ( ( unsigned long long ) Ptr );
}

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size )
{
  ThreadLocalTPUDeviceManager.CopyDeviceToDevice ( Dst, Src, Size );
}

} // namespace tpu
