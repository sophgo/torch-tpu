#include <vector>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <c10/util/Logging.h>
#include <c10/core/DeviceType.h>
#include <sys/time.h>
#include "TPUDeviceManager.h"

#ifdef BACKEND_SG2260
#include "TPUStream.h"
#include <sgdnn_api.h>
#include <tpuv7_rt.h>
namespace tpu
{
using Devptr = unsigned char *;

static inline size_t alignTo(size_t s, size_t n)
{
  return (s + n - 1) / n * n;
}

size_t getTPUAllocatorFreeDelay()
{
  const char *delay = getenv("TPU_ALLOCATOR_FREE_DELAY_IN_MS");
  if (!delay) return 0;
  return atoi(delay);
}

size_t getTPUAllocatorForcedTTL()
{
  const char *delay = getenv("TPU_ALLOCATOR_FORCED_TTL_IN_MS");
  if (!delay) return 0;
  return atoi(delay);
}

size_t getTPUAllocatorAlignSize(size_t defaultVal)
{
  const char *size = getenv("TPU_ALLOCATOR_ALIGN_SIZE");
  if (!size) return defaultVal;
  return atoi(size);
}

bool getEnableAllocatorReuse()
{
  const char *reuseEnabled = getenv("TPU_ALLOCATOR_REUSE");
  if (!reuseEnabled) return true;
  return atoi(reuseEnabled);
}

class TPUDeviceManager
{
private:
  struct MemToFree {
    void *ptr;
    tpuRtEvent_t event;
    tpuRtStream_t stream;
    uint64_t marked_timestamp = 0, freed_timestamp = 0;
  };

  struct MemInfo {
    size_t size;
  };

public:
  TPUDeviceManager() : init_flag_(false) {
    this->initialize();
  }

  ~TPUDeviceManager() {
    if (tpu_resource_t stream_ = c10_tpu::getDefaultTPUStream()) {
      sgdnnDeinitialize(stream_);
      tpuRtStreamSynchronize(stream_);
      tpuRtStreamDestroy(stream_);
    }
  }

  TPUMgrStatus Initialized() { return init_flag_ ? INIT_ALREADY : NOT_INIT;}

  TPUMgrStatus initialize()
  {
    if (init_flag_) return INIT_SUCCESS;

    tpuRtInit();

    int DeviceCount = 0;
    tpuRtStatus_t Status = tpuRtGetDeviceCount ( &DeviceCount );
    if (DeviceCount == 0) {
      std::cout << "Device Count:" << DeviceCount << "\n";
      DeviceCount = 1;

      char* size = getenv("OMPI_COMM_WORLD_SIZE");
      if (size == nullptr) {
          size = getenv("LOCAL_WORLD_SIZE");
      }
      if(size != nullptr) {
        DeviceCount = atoi(size);
      }
    }

    TORCH_CHECK ( Status == tpuRtSuccess, "Failed to get TPU device count" );
    if ( DeviceCount > 0 )
    {
      mutexes_ = std::vector<std::mutex>(DeviceCount);;
      free_tbds_.resize(DeviceCount);
      mem_info_.resize(DeviceCount);

      devices_init_ = std::vector<std::atomic<bool>> ( DeviceCount );
      device_count_ = DeviceCount;
      devices_init_[0] = 1; // tpuRtDeviceInit will default set idx = 0 device. that not a good idea.
    }
    SOPHON_LOG("TPU Device Manager init successfully\n");
    forced_ttl_ = getTPUAllocatorForcedTTL();
    free_delay_ = getTPUAllocatorFreeDelay();
    init_flag_ = true;
    return INIT_SUCCESS;
  }

  TPUMgrStatus InitDevice(int Index ){
    tpuRtStatus_t Status = tpuRtSetDevice( Index );
    TORCH_CHECK (Status == tpuRtSuccess, " sgSetDevice failed! Error Code : #", Status);
    return INIT_SUCCESS;
  }

  static TPUDeviceManager& GetInstance()
  {
    static TPUDeviceManager instance;
    return instance;
  }

  int GetDeviceCount() const
  {
    return device_count_;
  }

  size_t getSize(void *ptr, int i)
  {
    auto &mem_info = getMemInfo(i);
    if (mem_info.find(ptr) == mem_info.end())
    {
      return 0;
    }
    return mem_info[ptr].size;
  }

  inline static uint64_t gettime()
  {
    struct timeval t;
    gettimeofday ( &t, NULL );
    uint64_t now = (t.tv_sec * 1000000UL + t.tv_usec) / 1000UL;
    return now;
  }

  void *processFreeTBDs(size_t size, int index)
  {
    auto now = gettime();

    std::lock_guard<std::mutex> lock(getMutex(index));

    void *reusedPtr = nullptr;
    std::vector<void *> toRm;
    auto &freeTBDs = getFreeTBDs(index);
    static bool reuseEnabled = getEnableAllocatorReuse();
    for (auto &pair : freeTBDs)
    {
      auto &event = pair.second.event;
      auto &tbd = pair.second;
      if (tbd.freed_timestamp == 0)
      {
        if (forced_ttl_)
        {
            if (tbd.marked_timestamp + forced_ttl_ < now)
              tbd.freed_timestamp = now;
            else continue;
        } else {
            if (tpuRtEventQuery(event) != tpuRtDevnotready)
            {
              tbd.freed_timestamp = now;
            } else {
                continue;
            }
        }
      }

      if (reuseEnabled && !reusedPtr && getSize(tbd.ptr, index) == size)
      {
        // Reuse it
        toRm.push_back(tbd.ptr);
        reusedPtr = tbd.ptr;
        //std::cout << "Re-use " << reusedPtr << std::endl;
        break;
      }

      if (tbd.freed_timestamp + free_delay_ > now)
      {
        continue;
      }

      tpuRtFree(&tbd.ptr, NO_USE);
      // std::cout << "Free " << tbd.ptr << " of size " << getSize(tbd.ptr, index) << std::endl;
      getMemInfo(index).erase(tbd.ptr);
      toRm.push_back(tbd.ptr);
    }

    for (auto ptr : toRm)
    {
      auto &tbd = freeTBDs[ptr];
      if (!forced_ttl_)
        tpuRtEventFree(tbd.event, tbd.stream);
      freeTBDs.erase(ptr);
    }

#ifdef PROFILE_MEMORY_REUSE
    size_t sizeDefered = 0;
    size_t sizeDelayed = 0;
    for (auto &pair : freeTBDs)
    {
      auto &tbd = pair.second;
      if (tbd.freed_timestamp == 0)
        sizeDefered += getSize(tbd.ptr, index);
      else
        sizeDelayed += getSize(tbd.ptr, index);
    }
    std::cout << "===== Memory defered by TPU tasks " << sizeDefered
              << ". Memory delayed by reuse strategy " << sizeDelayed
              << std::endl;
#endif

    return reusedPtr;
  }

  void * Alloc ( size_t Size, int Index )
  {
    static size_t alignSize = getTPUAllocatorAlignSize(0x100000);
    Size = alignTo(Size, alignSize);

    auto reused = processFreeTBDs(Size, Index);
    if (reused)
      return (void*)UnifiedAddr((unsigned long long) reused, Index);

    if ( Size == 0 )
    {
      return nullptr;
    }
    Devptr dev_ptr;
    tpuRtStatus_t Status = tpuRtMalloc((void **)(&dev_ptr), Size, NO_USE);
    TORCH_CHECK ( Status == tpuRtSuccess, "Failed to allocate memory on TPU device #", Index, " size = ", Size, "bytes" );

    MemInfo info;
    info.size = Size;
    auto ptr = ( void * ) UnifiedAddr((unsigned long long) dev_ptr, Index);
    std::lock_guard<std::mutex> lock(getMutex(Index));
    getMemInfo(Index)[dev_ptr] = info;

    // std::cout << "Alloc " << ptr << " of size " << Size << std::endl;

    return ptr;
  }

  void Free ( void * Ptr, int Index )
  {
    if ( Ptr == nullptr )
    {
      return;
    }

    MemToFree tbd;
    tbd.stream = c10_tpu::getCurrentTPUStream();
    std::lock_guard<std::mutex> lock(getMutex(Index));

    auto &memInfo = getMemInfo(Index);
    if (memInfo.find(Ptr) == memInfo.end())
      return;

    auto &freeTBDs = getFreeTBDs(Index);
    if (freeTBDs.find(Ptr) != freeTBDs.end())
      return;

    tbd.ptr = Ptr;
    if (forced_ttl_)
      tbd.marked_timestamp = gettime();

    auto status = tpuRtEventCreate(&tbd.event);
    TORCH_CHECK (status == tpuRtSuccess, "Failed to create event");
    status = tpuRtEventRecord(tbd.event, tbd.stream);
    TORCH_CHECK (status == tpuRtSuccess, "Failed to record stream");
    freeTBDs[Ptr] = tbd;
  }

  void CopyHostToDevice ( void * Dst, const void * Src, size_t Size, int Index, bool non_blocking )
  {
    tpuRtStatus_t Status;
    auto stream = c10_tpu::getDefaultTPUStream();
    if (!non_blocking) {
      Status = tpuRtMemcpyS2DAsync( Dst, Src, Size, stream );
      tpuRtStreamSynchronize(stream);
    } else {
      // struct timeval timer, end, timer1;
      // gettimeofday ( &timer1, NULL );
      // gettimeofday ( &timer, NULL );
      Status = tpuRtMemcpyS2DAsync( Dst, Src, Size, stream );
      // gettimeofday ( &end, NULL );
      // std::cout << " tpuRtMemcpyS2DAsync getstream time : " << ( timer.tv_sec - timer1.tv_sec ) * 1000000UL + ( timer.tv_usec - timer1.tv_usec ) << "us\n";
      // std::cout << " sgMemcpyS2SAsync  time : " << ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec ) << "us\n";
    }
    TORCH_CHECK ( Status == tpuRtSuccess, "Failed to copy memory from host to TPU device #", Index, " size = ", Size, "bytes" );
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size, int Index, bool non_blocking )
  {
    tpuRtStatus_t Status;
    auto stream = c10_tpu::getDefaultTPUStream();
    if(!non_blocking) {
      Status = tpuRtMemcpyD2SAsync( Dst, Src, Size, stream );
      tpuRtStreamSynchronize(stream);
    } else {
      // struct timeval timer1, timer, end;
      // gettimeofday ( &timer1, NULL );
      // gettimeofday ( &timer, NULL );
      Status = tpuRtMemcpyD2SAsync( Dst, Src, Size, stream );
      // gettimeofday ( &end, NULL );
      // std::cout << " tpuRtMemcpyD2SAsync getstream time : " << ( timer.tv_sec - timer1.tv_sec ) * 1000000UL + ( timer.tv_usec - timer1.tv_usec ) << "us\n";
      // std::cout << " tpuRtMemcpyD2SAsync  time : " << ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec ) << "us\n";
    }
    TORCH_CHECK ( Status == tpuRtSuccess, "Failed to copy memory from TPU device #", Index, " to host size = ", Size, "bytes" );
  }

  void CopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, int Index, bool non_blocking )
  {
    // if (!non_blocking) {
    //   // tpuRtStatus_t Status = sgMemcpyD2D( Dst, Src, Size);
    // } else {
    //   //tpuRtStatus_t Status = sgMemcpyD2DAsync( Dst, Src, Size, stream );
    // }
    auto stream = c10_tpu::getDefaultTPUStream();
    SgdnnTensor_t input = {.addr = (unsigned long long)Src, .dim = 1, .dtype = SGDNN_DTYPE_INT8};
    SgdnnTensor_t output = {.addr = (unsigned long long)Dst, .dim = 1, .dtype = SGDNN_DTYPE_INT8};
    input.shape[0] = output.shape[0] = Size;
    input.stride[0] = output.stride[0] = 1;
    tpuRtStatus_t Status = sgdnnStridedCopy(stream, input, output, non_blocking);
    TORCH_CHECK (Status == tpuRtSuccess, "TPU device #", Index, "D2D failed! Error Code : #", Status);
  }

private:
  std::mutex& getMutex(int i) { return mutexes_[i]; }
  std::unordered_map<void *, MemInfo>& getMemInfo(int i) { return mem_info_[i]; }
  std::unordered_map<void *, MemToFree> &getFreeTBDs(int i) { return free_tbds_[i]; }

  std::vector<std::mutex> mutexes_;
  std::vector<std::unordered_map<void *, MemToFree>> free_tbds_;
  std::vector<std::unordered_map<void *, MemInfo>> mem_info_;

  int device_count_;
  size_t free_delay_, forced_ttl_;
  std::atomic<bool> init_flag_;
  std::vector<std::atomic<bool>> devices_init_;

  TPUDeviceManager(const TPUDeviceManager&) = delete;
  TPUDeviceManager& operator=(const TPUDeviceManager&) = delete;
};

TPUDeviceManager* TPUGetInstance(){
  return &TPUDeviceManager::GetInstance();
}

TPUMgrStatus InitTPUMgr()
{
  return TPUGetInstance()->initialize();
}

TPUMgrStatus DestoryTpuMgr()
{
   delete TPUGetInstance();
   return DESTORY_SUCCESS;
}

TPUMgrStatus IsTPUMgrInited()
{
  return TPUGetInstance()->Initialized();
}

TPUMgrStatus TPUDeviceInitialize( int Index ) {
  InitTPUMgr();
  return TPUDeviceManager::GetInstance().InitDevice(Index);
}

int TPUGetDeviceCount ( void )
{
  return TPUDeviceManager::GetInstance().GetDeviceCount();
}

int TPUGetDeviceIndex ( void )
{
  int DevIndex;
  tpuRtStatus_t Status = tpuRtGetDevice( &DevIndex );
  if (Status == tpuRtErrorNoDevice) {
    InitTPUMgr();
    DevIndex = 0;
  } else
  {
    TORCH_CHECK (Status == tpuRtSuccess, " sgGetDevice failed! Error Code : #", Status);
  }
  return DevIndex;
}

void TPUSetDeviceIndex ( int Index )
{
  tpuRtStatus_t Status = tpuRtSetDevice( Index );
  TORCH_CHECK (Status == tpuRtSuccess, " sgSetDevice failed! Error Code : #", Status);
}

void * TPUAlloc ( size_t Size )
{
  return TPUDeviceManager::GetInstance().Alloc ( Size, TPUGetDeviceIndex() );
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


void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size,  bool non_blocking)
{
  if ( Size == 0 ) return;
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Dst);
  unsigned long long dst_ptr = GetAddrByUnifiedAddr((unsigned long long)Dst);
  TPUDeviceManager::GetInstance().CopyHostToDevice ( (void*)dst_ptr, Src, Size, dev_index, non_blocking );
}

void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size, bool non_blocking )
{
  if ( Size == 0 ) return;
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Src);
  unsigned long long src_ptr = GetAddrByUnifiedAddr((unsigned long long)Src);
  TPUDeviceManager::GetInstance().CopyDeviceToHost ( Dst, (void*)src_ptr, Size, dev_index, non_blocking );
}

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking )
{
  unsigned long long src_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Src);
  unsigned long long src_ptr = GetAddrByUnifiedAddr((unsigned long long)Src);
  unsigned long long dst_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Dst);
  unsigned long long dst_ptr = GetAddrByUnifiedAddr((unsigned long long)Dst);
  TORCH_CHECK ( dst_index == src_index, "D2D copy must in same device");
  TPUDeviceManager::GetInstance().CopyDeviceToDevice ( (void*)dst_ptr, (void*)src_ptr, Size, dst_index, non_blocking );
}

tpuRtStream_t TPUGetDeviceResource ( void ) {
  auto stream = c10_tpu::getCurrentTPUStream();
  tpudnnFlush(stream);
  return stream;
}

} // namespace tpu

#endif //BACKEND_SG2260
