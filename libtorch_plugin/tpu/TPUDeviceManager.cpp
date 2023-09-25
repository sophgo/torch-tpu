#include <vector>
#include <unordered_map>
#include <mutex>
#include <c10/util/Logging.h>
#include <TPUDeviceManager.h>
#include <sgdnn_api.h>
#include <TPUTorchUtils.h>

#include <iostream>

namespace tpu
{
#ifdef SHOW_MALLOC_INFO
static unsigned long long mem_in_use = 0;
#endif

class TPUDeviceManager
{
private:
  TPUDeviceManager()
  {
    int DeviceCount = 0;
    bm_status_t Status = bm_dev_getcount ( &DeviceCount );
    char * MaxDeviceCountStr = nullptr;
    if ( ( MaxDeviceCountStr = getenv ( "SG_MAX_DEVICE_COUNT" ) ) )
    {
      int MaxDeviceCount = atoi ( MaxDeviceCountStr );
      TORCH_CHECK ( MaxDeviceCount > 0 );
      if ( DeviceCount > MaxDeviceCount )
      {
        DeviceCount = MaxDeviceCount;
      }
    }
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to get TPU device count" );
    if ( DeviceCount > 0 )
    {
      Handles_.resize ( DeviceCount, nullptr );
      for ( int i = 0; i < DeviceCount; ++i )
      {
        Status = bm_dev_request ( &Handles_[i], i );
        TORCH_CHECK ( Status == BM_SUCCESS, "Failed to request tpu device #", i );
        Status = sgdnnInitialize ( Handles_[i] );
        TORCH_CHECK ( Status == BM_SUCCESS, "Failed to initialize SGDNN on tpu device #", i );
      }
      Mutexes_ = std::vector<std::mutex> ( DeviceCount );
      AddrMemMaps_ = std::vector<std::unordered_map<unsigned long long, bm_device_mem_t>> ( DeviceCount );
    }
  }

  ~TPUDeviceManager()
  {
    for ( auto it : Handles_ )
    {
      sgdnnDeinitialize ( it );
      bm_dev_free ( it );
    }
  }

  static TPUDeviceManager * instance_;

public:

  static TPUDeviceManager & GetInstance()
  {
    if ( instance_ == nullptr )
    {
      instance_ = new TPUDeviceManager();
    }
    return *instance_;
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

  void * Alloc ( size_t Size, int Index )
  {
    Mutexes_[Index].lock();
    if ( Size == 0 )
    {
      Mutexes_[Index].unlock();
      return nullptr;
    }
    TORCH_CHECK ( Size < ( 1UL << 32 ), "TPU only allows to allocate memory with size smaller than (2^32) bytes" );
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    bm_device_mem_t Mem;
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t Status = bm_malloc_device_byte ( Handle, &Mem, ( unsigned int ) Size );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::MALLOC, timer.ElapsedUS() );
#endif
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to allocate memory on TPU device #", Index, " size = ", Size, "bytes" );
    unsigned long long Addr = bm_mem_get_device_addr ( Mem );
    AddrMemMaps_[Index].emplace ( Addr, Mem );
#ifdef SHOW_MALLOC_INFO
    static int malloc_num = 0;
    malloc_num++;
    mem_in_use += ( unsigned int ) Size;
    std::cout << "[Malloc] id = " << malloc_num << ", size = " << Size  << " bytes"
              << ", Current Mem = " << ( mem_in_use >> 20 ) << "MB" << std::endl;
#endif
#ifdef SHOW_INFO
    std::cout << "Alloc addr = " << ( void * ) Addr << " size = " << Size << std::endl;
    std::cout << "====================================" << std::endl;
    for ( auto iter : AddrMemMaps_[Index] )
    {
      std::cout << ( void * ) iter.first << " ";
    }
    std::cout << std::endl;
    std::cout << "====================================" << std::endl;
#endif
    Mutexes_[Index].unlock();
    return ( void * ) Addr;
  }

  void Free ( void * Ptr, int Index )
  {
    Mutexes_[Index].lock();
    if ( Ptr == nullptr )
    {
      Mutexes_[Index].unlock();
      return;
    }
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    auto Iter = AddrMemMaps_[Index].find ( ( unsigned long long ) Ptr );
    TORCH_CHECK ( Iter != AddrMemMaps_[Index].end(), "Memory of address = ", Ptr, " is not found" );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_free_device ( Handle, Iter->second );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::FREE, timer.ElapsedUS() );
#endif
#ifdef SHOW_MALLOC_INFO
    static int free_num = 0;
    free_num++;
    mem_in_use -= Iter->second.size;
    std::cout << "[Free] id = " << free_num << ", size = " << Iter->second.size << " bytes"
              << ", Current Mem = " << ( mem_in_use >> 20 ) << "MB" << std::endl;
#endif
    AddrMemMaps_[Index].erase ( Iter );
#ifdef SHOW_INFO
    std::cout << "Free addr = " << Ptr << std::endl;
    std::cout << "====================================" << std::endl;
    for ( auto iter : AddrMemMaps_[Index] )
    {
      std::cout << ( void * ) iter.first << " ";
    }
    std::cout << std::endl;
    std::cout << "====================================" << std::endl;
#endif
    Mutexes_[Index].unlock();
  }

  void CopyHostToDevice ( void * Dst, const void * Src, size_t Size, int Index )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Host = " << Src << " to Device = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    bm_device_mem_t DstMem = bm_mem_from_device ( ( unsigned long long ) Dst, Size );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t Status = bm_memcpy_s2d ( Handle, DstMem, ( void * ) Src );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from host to TPU device #", Index, " size = ", Size, "bytes" );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CDMA_S2D, timer.ElapsedUS() );
#endif
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size, int Index )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Device = " << Src << " to Host = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    bm_device_mem_t SrcMem = bm_mem_from_device ( ( unsigned long long ) Src, Size );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t Status = bm_memcpy_d2s ( Handle, Dst, SrcMem );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from TPU device #", Index, " to host size = ", Size, "bytes" );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CDMA_D2S, timer.ElapsedUS() );
#endif
  }

  void CopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, int Index )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Device = " << Src << " to Device = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    auto DstMem = bm_mem_from_device ( ( unsigned long long ) Dst, Size );
    auto SrcMem = bm_mem_from_device ( ( unsigned long long ) Src, Size );
    bm_status_t Status = BM_SUCCESS;
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    Status = bm_memcpy_d2d_byte ( Handle, DstMem, 0, SrcMem, 0, Size );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::COPY, timer.ElapsedUS() );
#endif
  }

private:
  std::vector<bm_handle_t> Handles_;
  std::vector<std::mutex> Mutexes_;
  std::vector<std::unordered_map<unsigned long long, bm_device_mem_t>> AddrMemMaps_;
};

TPUDeviceManager * TPUDeviceManager::instance_ = nullptr;

static thread_local int kIndex = 0;

int TPUGetDeviceCount ( void )
{
  return TPUDeviceManager::GetInstance().GetDeviceCount();
}

int TPUGetDeviceIndex ( void )
{
  return kIndex;
}

void TPUSetDeviceIndex ( int Index )
{
  if (Index == -1)
    kIndex = 0;
  else
    kIndex = Index;
}

bm_handle_t TPUGetDeviceHandle()
{
  return TPUDeviceManager::GetInstance().GetDeviceHandle ( kIndex );
}

void * TPUAlloc ( size_t Size )
{
  return TPUDeviceManager::GetInstance().Alloc ( Size, kIndex );
}

void * TPUAlloc ( size_t Size, int Index )
{
  return TPUDeviceManager::GetInstance().Alloc ( Size, Index );
}

void TPUFree ( void * Ptr )
{
  TPUDeviceManager::GetInstance().Free ( Ptr, kIndex );
}

void TPUFree ( void * Ptr, int Index )
{
  TPUDeviceManager::GetInstance().Free ( Ptr, Index );
}

void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size )
{
  TPUDeviceManager::GetInstance().CopyHostToDevice ( Dst, Src, Size, kIndex );
}

void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size )
{
  TPUDeviceManager::GetInstance().CopyDeviceToHost ( Dst, Src, Size, kIndex );
}

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size )
{
  TPUDeviceManager::GetInstance().CopyDeviceToDevice ( Dst, Src, Size, kIndex );
}

} // namespace tpu
