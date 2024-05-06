
#include <vector>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <thread>
#include <atomic>

#include <c10/util/Logging.h>
#include <c10/core/DeviceType.h>
#include "TPUDeviceManager.h"

#ifdef BACKEND_1684X
#include <sgdnn_api.h>
#include "TPUTorchUtils.h"

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
      Mutexes_ = std::vector<std::mutex> ( DeviceCount );
      AddrMemMaps_ = std::vector<std::unordered_map<unsigned long long, sg_device_mem_t>> ( DeviceCount );
    }
  }

  ~TPUDeviceManager()
  {
    for ( auto it : Handles_ )
    {
      if (it != nullptr){
        sgdnnDeinitialize ( it );
        bm_dev_free ( it );
      }
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

#ifdef SHOW_MALLOC_INFO
    static int malloc_num = 0;
    malloc_num++;
    mem_in_use += ( unsigned int ) Size;
    std::cout << "[Malloc] id = " << malloc_num << ", size = " << Size  << " bytes"
              << ", Current Mem = " << ( mem_in_use >> 20 ) << "MB" << std::endl;
#endif

    // TORCH_CHECK ( Size < ( 1UL << 32 ), "TPU only allows to allocate memory with size smaller than (2^32) bytes" );
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    sg_device_mem_t Mem;
    bm_status_t Status = sg_malloc_device_byte ( Handle, &Mem, Size );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to allocate memory on TPU device #", Index, " size = ", Size, "bytes" );
    unsigned long long Addr = sg_mem_get_device_addr ( Mem );
    AddrMemMaps_[Index].emplace ( Addr, Mem );

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
    return ( void * ) UnifiedAddr(Addr, Index);
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
    sg_free_device ( Handle, Iter->second );

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
    sg_device_mem_t DstMem = sg_mem_from_device ( ( unsigned long long ) Dst, Size );
    bm_status_t Status = sg_memcpy_s2d ( Handle, DstMem, ( void * ) Src );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from host to TPU device #", Index, " size = ", Size, "bytes" );
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size, int Index )
  {
#ifdef SHOW_INFO
    std::cout << "Copy Device = " << Src << " to Host = " << Dst << " Size = " << Size << std::endl;
#endif
    bm_handle_t Handle = GetDeviceHandle ( Index );
    TORCH_CHECK ( Handle != nullptr, "TPU handle of device #", Index, " is null" );
    sg_device_mem_t SrcMem = sg_mem_from_device ( ( unsigned long long ) Src, Size );
    bm_status_t Status = sg_memcpy_d2s ( Handle, Dst, SrcMem );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to copy memory from TPU device #", Index, " to host size = ", Size, "bytes" );
  }

void CopyDeviceToDevice(void* Dst, const void* Src, size_t Size, int Index)
{
#ifdef SHOW_INFO
  std::cout << "Copy Device = " << Src << " to Device = " << Dst << " Size = " << Size << std::endl;
#endif
  bm_handle_t Handle = GetDeviceHandle(Index);
  TORCH_CHECK(Handle != nullptr, "TPU handle of device #", Index, " is null");
  // no sg_memcpy_d2d, use multiple bm_memcpy_d2d
  const size_t chunkSize = __INT32_MAX__ - 65535; // avoid tpu_gdma_cpy_S2S overflow
  const size_t numChunks = (Size + chunkSize - 1) / chunkSize;

  for (size_t i = 0; i < numChunks; ++i)
  {
    size_t chunkOffset = i * chunkSize;
    size_t chunkCopySize = std::min(chunkSize, Size - chunkOffset);

    auto DstMem = bm_mem_from_device((unsigned long long)Dst + chunkOffset, chunkCopySize);
    auto SrcMem = bm_mem_from_device((unsigned long long)Src + chunkOffset, chunkCopySize);

    bm_status_t Status = BM_SUCCESS;
    Status = bm_memcpy_d2d_byte ( Handle, DstMem, 0, SrcMem, 0, Size );
    TORCH_CHECK (Status == BM_SUCCESS, "D2D failed! Error Code : #", Status );
  }
}

  bm_handle_t& get_device_handle(int i){ return Handles_[i];}
  std::mutex& get_handle_mutex(int i) { return Mutexes_[i]; }
  std::unordered_map<unsigned long long, sg_device_mem_t>& getAddrMems(int i) { return AddrMemMaps_[i]; }

private:
  std::vector<bm_handle_t> Handles_;
  std::vector<std::mutex> Mutexes_;
  std::vector<std::unordered_map<unsigned long long, sg_device_mem_t>> AddrMemMaps_;
};

TPUDeviceManager * TPUDeviceManager::instance_ = nullptr;

static thread_local int kIndex = 0;

TPUDeviceManager* TPUGetInstance(){
  return &TPUDeviceManager::GetInstance();
}

TPUMgrStatus InitTPUMgr() { TPUSetDeviceIndex(0); return INIT_SUCCESS; }
TPUMgrStatus DestoryTpuMgr() { return DESTORY_SUCCESS;}
TPUMgrStatus IsTPUMgrInited() { return INIT_ALREADY; }
TPUMgrStatus TPUDeviceInitialize( int Index) { TPUSetDeviceIndex(Index); return INIT_SUCCESS;}

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
  kIndex = Index;
  if (kIndex == -1) kIndex = 0;

  TPUDeviceManager::GetInstance().get_handle_mutex(kIndex).lock();
  if ( kIndex >= TPUGetDeviceCount() || kIndex < 0){
    TORCH_CHECK ( false, "Failed to request tpu device #", kIndex );
  }
  bm_handle_t& handle = TPUDeviceManager::GetInstance().get_device_handle(kIndex);
  if (handle == nullptr){
    auto Status = bm_dev_request ( &handle, kIndex );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to request tpu device #", kIndex );
    Status = sgdnnInitialize ( handle );
    TORCH_CHECK ( Status == BM_SUCCESS, "Failed to initialize SGDNN on tpu device #", kIndex );
  }
  TPUDeviceManager::GetInstance().get_handle_mutex(kIndex).unlock();
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
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Ptr);
  unsigned long long data_ptr = GetAddrByUnifiedAddr((unsigned long long)Ptr);
  TPUDeviceManager::GetInstance().Free ( (void *)data_ptr, dev_index );
}


void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking /*= false */ )
{
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Dst);
  unsigned long long dst_ptr = GetAddrByUnifiedAddr((unsigned long long)Dst);
  TPUDeviceManager::GetInstance().CopyHostToDevice ( (void*)dst_ptr, Src, Size, dev_index );
}

void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size, bool non_blocking /*= false */ )
{
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Src);
  unsigned long long src_ptr = GetAddrByUnifiedAddr((unsigned long long)Src);
  TPUDeviceManager::GetInstance().CopyDeviceToHost ( Dst, (void*)src_ptr, Size, dev_index );
}

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking /*= false */ )
{
  unsigned long long src_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Src);
  unsigned long long src_ptr = GetAddrByUnifiedAddr((unsigned long long)Src);
  unsigned long long dst_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Dst);
  unsigned long long dst_ptr = GetAddrByUnifiedAddr((unsigned long long)Dst);
  TORCH_CHECK ( dst_index == src_index, "D2D copy must in same device");
  TPUDeviceManager::GetInstance().CopyDeviceToDevice ( (void*)dst_ptr, (void*)src_ptr, Size, dst_index );
}

} // namespace tpu

#endif // BACKEND_1684X