#include <vector>
#include <unordered_map>
#include <mutex>
#include <c10/util/Logging.h>
#include <TPUDeviceManager.h>
#include <sgdnn_api.h>
#include <TPUTorchUtils.h>

#include <iostream>

//#define SHOW_INFO
#define TPU_OP_TIMING

namespace tpu
{

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
        tpu_module_init ( Handles_[i] );
      }
      Index_ = 0;
    }
    else
    {
      Index_ = -1;
    }
    Mutexes_ = std::vector<std::mutex> ( DeviceCount );
    AddrMemMaps_ = std::vector<std::unordered_map<unsigned long long, bm_device_mem_t>> ( DeviceCount );
  }

  ~TPUDeviceManager()
  {
    for ( auto it : Handles_ )
    {
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

  void * Alloc ( size_t Size )
  {
    Mutexes_[Index_].lock();
    if ( Size == 0 )
    {
      Mutexes_[Index_].unlock();
      return nullptr;
    }
    TORCH_CHECK ( Size < ( 1UL << 32 ), "TPU only allows to allocate memory with size smaller than (2^32) bytes" );
    bm_handle_t Handle = GetDeviceHandle();
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    bm_device_mem_t Mem;
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t Status = bm_malloc_device_byte ( Handle, &Mem, ( unsigned int ) Size );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::MALLOC, timer.ElapsedUS() );
#endif
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to allocate memory on TPU device #", GetDeviceIndex(), " size = ", Size, "bytes" );
    unsigned long long Addr = bm_mem_get_device_addr ( Mem );
    AddrMemMaps_[Index_].emplace ( Addr, Mem );
#ifdef SHOW_INFO
    std::cout << "Alloc addr = " << ( void * ) Addr << " size = " << Size << std::endl;
    std::cout << "====================================" << std::endl;
    for ( auto iter : AddrMemMaps_[Index_] )
    {
      std::cout << ( void * ) iter.first << " ";
    }
    std::cout << std::endl;
    std::cout << "====================================" << std::endl;
#endif
    Mutexes_[Index_].unlock();
    return ( void * ) Addr;
  }

  void Free ( void * Ptr )
  {
    Mutexes_[Index_].lock();
    if ( Ptr == nullptr )
    {
      Mutexes_[Index_].unlock();
      return;
    }
    bm_handle_t Handle = GetDeviceHandle();
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    auto Iter = AddrMemMaps_[Index_].find ( ( unsigned long long ) Ptr );
    TORCH_CHECK ( Iter != AddrMemMaps_[Index_].end(), "Memory of address = ", Ptr, " is not found" );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_free_device ( Handle, Iter->second );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::FREE, timer.ElapsedUS() );
#endif
    AddrMemMaps_[Index_].erase ( Iter );
#ifdef SHOW_INFO
    std::cout << "Free addr = " << Ptr << std::endl;
    std::cout << "====================================" << std::endl;
    for ( auto iter : AddrMemMaps_[Index_] )
    {
      std::cout << ( void * ) iter.first << " ";
    }
    std::cout << std::endl;
    std::cout << "====================================" << std::endl;
#endif
    Mutexes_[Index_].unlock();
  }

  void CopyHostToDevice ( void * Dst, const void * Src, size_t Size )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Host = " << Src << " to Device = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle();
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    bm_device_mem_t DstMem = bm_mem_from_device ( ( unsigned long long ) Dst, Size );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t Status = bm_memcpy_s2d ( Handle, DstMem, ( void * ) Src );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from host to TPU device #", Index_, " size = ", Size, "bytes" );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CDMA_S2D, timer.ElapsedUS() );
#endif
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Device = " << Src << " to Host = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle();
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
    bm_device_mem_t SrcMem = bm_mem_from_device ( ( unsigned long long ) Src, Size );
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t Status = bm_memcpy_d2s ( Handle, Dst, SrcMem );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from TPU device #", Index_, " to host size = ", Size, "bytes" );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CDMA_D2S, timer.ElapsedUS() );
#endif
  }

  void CopyDeviceToDevice ( void * Dst, const void * Src, size_t Size )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Device = " << Src << " to Device = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle();
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", GetDeviceIndex(), " is null" );
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
  int Index_;
  std::vector<std::mutex> Mutexes_;
  std::vector<std::unordered_map<unsigned long long, bm_device_mem_t>> AddrMemMaps_;
};

TPUDeviceManager * TPUDeviceManager::instance_ = nullptr;

int TPUGetDeviceCount ( void )
{
  return TPUDeviceManager::GetInstance().GetDeviceCount();
}

int TPUGetDeviceIndex ( void )
{
  return TPUDeviceManager::GetInstance().GetDeviceIndex();
}

void TPUSetDeviceIndex ( int Index )
{
  TPUDeviceManager::GetInstance().SetDeviceIndex ( Index );
}

bm_handle_t TPUGetDeviceHandle()
{
  return TPUDeviceManager::GetInstance().GetDeviceHandle();
}

void * TPUAlloc ( size_t Size )
{
  return TPUDeviceManager::GetInstance().Alloc ( Size );
}

void TPUFree ( void * Ptr )
{
  TPUDeviceManager::GetInstance().Free ( Ptr );
}

void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size )
{
  TPUDeviceManager::GetInstance().CopyHostToDevice ( Dst, Src, Size );
}

void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size )
{
  TPUDeviceManager::GetInstance().CopyDeviceToHost ( Dst, Src, Size );
}

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size )
{
  TPUDeviceManager::GetInstance().CopyDeviceToDevice ( Dst, Src, Size );
}

} // namespace tpu
