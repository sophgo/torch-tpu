#include <unistd.h>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <c10/util/Logging.h>
#include <c10/core/DeviceType.h>
#include <sys/time.h>
#include "TPUDeviceManager.h"
#include "TPUStream.h"
#include <tpu_runtime_api.h>
#include "KernelManager.hpp"
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

int getVisibleDeviceCount()
{
  int DeviceCount = 0;
  const char *visibe_tpu_num = getenv("TPU_VISIBLE_NUM");
  if (visibe_tpu_num) DeviceCount = atoi(visibe_tpu_num);
  return DeviceCount;
}

static std::shared_ptr<TPUDeviceManager> devMgrPtr;
static int VISIBLE_DEVICE_NUM = getVisibleDeviceCount();


class TPUDeviceManager
{
private:
  struct MemToFree {
    void *ptr;
    tpuEvent_t event;
    tpuStream_t stream;
    uint64_t marked_timestamp = 0, freed_timestamp = 0;
  };

  struct MemInfo {
    size_t size;
  };
  bool disable_ins_cache = false;
public:
  TPUDeviceManager() : init_flag_(false) {
    this->initialize();
    disable_ins_cache = std::getenv("DISABLE_CACHE") != nullptr;
  }

  ~TPUDeviceManager() {
    int device = -1;
    auto ret = tpuGetDevice(&device);
    if (device >= 0 && ret == tpuSuccess) {
      std::lock_guard<std::mutex> lock(mutexes_[device]);
      for (const auto& [key, value] : mem_info_[device]) {
        tpuFree(key);
      }
    }
  }

  TPUMgrStatus Initialized() { return init_flag_ ? INIT_ALREADY : NOT_INIT;}

  TPUMgrStatus initialize()
  {
    if (init_flag_) return INIT_SUCCESS;

    tpuInit();

    int DeviceCount = 0;
    tpuError_t Status = tpuGetDeviceCount ( &DeviceCount );
    TORCH_CHECK ( Status == tpuSuccess, "Failed to get TPU device count" );
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


    if ( DeviceCount > 0 )
    {
      mutexes_ = std::vector<std::mutex>(DeviceCount);;
      free_tbds_.resize(DeviceCount);
      mem_info_.resize(DeviceCount);
      alloc_current_ = std::vector<size_t> ( DeviceCount, 0 );
      alloc_peak_ = std::vector<size_t> ( DeviceCount, 0 );

      devices_init_ = std::vector<std::atomic<bool>> ( DeviceCount );
      device_count_ = DeviceCount;
      devices_init_[0] = 1; // tpuRtDeviceInit will default set idx = 0 device. that not a good idea.
    }
    LOG( INFO ) << "TPU Device Manager init successfully\n";
#if 0
    // for multi-chip debug
    const char *rankStr = getenv("LOCAL_RANK");
    int rank = 0;
    if (rankStr)
        rank = atoi(rankStr);
    if (rank == 0)
    {
        volatile int i = 0;
        printf("PID %d ready for attach\n", getpid());
        fflush(stdout);
        while (0 == i);
    }
#endif
    forced_ttl_ = getTPUAllocatorForcedTTL();
    free_delay_ = getTPUAllocatorFreeDelay();
    init_flag_ = true;
    return INIT_SUCCESS;
  }

  TPUMgrStatus InitDevice(int Index ){
    auto Status = tpuSetDevice( Index );
    TORCH_CHECK (Status == tpuSuccess, " sgSetDevice failed! Error Code : #", Status);
    return INIT_SUCCESS;
  }

  static TPUDeviceManager& GetInstance()
  {
    static std::once_flag flag;
    std::call_once(
      flag,
      [&](){
        devMgrPtr = std::make_shared<TPUDeviceManager>();
      });
    return *devMgrPtr;
  }

  static void DeleteInstance()
  {
    devMgrPtr.reset();
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
            if (tpuEventQuery(event) == tpuSuccess)
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
      if (!disable_ins_cache){
        c10_tpu::getCurrentTPUStream().synchronize();
      }
      tpuFree(tbd.ptr);
      // std::cout << "Free " << tbd.ptr << " of size " << getSize(tbd.ptr, index) << std::endl;
      getMemInfo(index).erase(tbd.ptr);
      toRm.push_back(tbd.ptr);
    }

    for (auto ptr : toRm)
    {
      auto &tbd = freeTBDs[ptr];
      if (!forced_ttl_)
        tpuEventDestroy(tbd.event, tbd.stream);
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
    {
      std::lock_guard<std::mutex> lock(getMutex(Index));
      alloc_current_[Index] += Size;
      alloc_peak_[Index] = std::max(alloc_peak_[Index], alloc_current_[Index]);
      return (void*)UnifiedAddr((unsigned long long) reused, Index);
    }

    if ( Size == 0 )
    {
      return nullptr;
    }
    Devptr dev_ptr;

    // Try to allocate memory, if failed, free all reserved memory and retry once
    auto tryAllocate = [&]() -> tpuError_t{
      return tpuMalloc((void **)(&dev_ptr), Size);
    };

    tpuError_t Status = tryAllocate();
    if (Status != tpuSuccess) {
      std::cout << "First allocation failed, trying to free cache and retry. Device: #" << Index << " Size: " << Size << "bytes" << std::endl;
      // c10_tpu::getCurrentTPUStream().synchronize(); // sync for free memory but reduce performance
      EmptyCache(Index);
      Status = tryAllocate();
    }
    TORCH_CHECK(Status == tpuSuccess, "Failed to allocate memory on TPU device #", Index, " size = ", Size, " bytes. ",
                                      "If OOM, please try lowering the environment variable 'TPU_ALLOCATOR_FREE_DELAY_IN_MS' and current value is ", free_delay_, "ms. ");

    MemInfo info;
    info.size = Size;
    auto ptr = ( void * ) UnifiedAddr((unsigned long long) dev_ptr, Index);
    std::lock_guard<std::mutex> lock(getMutex(Index));
    getMemInfo(Index)[dev_ptr] = info;
    alloc_current_[Index] += Size;
    alloc_peak_[Index] = std::max(alloc_peak_[Index], alloc_current_[Index]);

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
    tbd.stream = c10_tpu::getCurrentTPUStream().stream();
    std::lock_guard<std::mutex> lock(getMutex(Index));

    auto &memInfo = getMemInfo(Index);
    if (memInfo.find(Ptr) == memInfo.end())
      return;

    alloc_current_[Index] -= memInfo[Ptr].size;

    auto &freeTBDs = getFreeTBDs(Index);
    if (freeTBDs.find(Ptr) != freeTBDs.end())
      return;

    tbd.ptr = Ptr;
    if (forced_ttl_)
      tbd.marked_timestamp = gettime();

    auto status = tpuEventCreate(&tbd.event);
    TORCH_CHECK (status == tpuSuccess, "Failed to create event");
    status = tpuEventRecord(tbd.event, tbd.stream);
    TORCH_CHECK (status == tpuSuccess, "Failed to record stream");
    // std::cout << "[Free] event = "<< tbd.event << "Dataptr = " << tbd.ptr << " of size " << getSize(tbd.ptr, index) << std::endl;
    freeTBDs[Ptr] = tbd;
  }

  void EmptyCache( int Index )
  {
    std::lock_guard<std::mutex> lock(getMutex(Index));
    auto &freeTBDs = getFreeTBDs(Index);
    std::vector<void *> toRm;
    for (auto &pair : freeTBDs)
    {
      auto &event = pair.second.event;
      auto &tbd = pair.second;
      if (tbd.freed_timestamp != 0 || tpuEventQuery(event) == tpuSuccess)
      {
        // std::cout << "[EmptyCache] event = "<< event <<"Dataptr = " << tbd.ptr << " of size " << getSize(tbd.ptr, index) << std::endl;
        tpuFree(tbd.ptr);
        getMemInfo(Index).erase(tbd.ptr);
        toRm.push_back(tbd.ptr);
      } else {
          continue;
      }
    }
    for (auto ptr : toRm)
    {
      auto &tbd = freeTBDs[ptr];
      tpuEventDestroy(tbd.event, tbd.stream);
      freeTBDs.erase(ptr);
    }
  }

  void StartCache( int Index )
  {
    std::lock_guard<std::mutex> lock(getMutex(Index));
    free_delay_ = getTPUAllocatorFreeDelay();
  }

  void StopCache( int Index )
  {
    std::lock_guard<std::mutex> lock(getMutex(Index));
    free_delay_ = 0;
  }

  void CopyHostToDevice ( void * Dst, const void * Src, size_t Size, int Index, bool non_blocking )
  {
    tpuError_t Status;
    auto stream = c10_tpu::getDefaultTPUStream();
    tpudnnFlush(stream);
    tpuStreamSynchronize(stream.stream());
    if (!non_blocking) {
      Status = tpuMemcpyH2D(Dst, Src, Size);
    } else {
      // struct timeval timer, end, timer1;
      // gettimeofday ( &timer1, NULL );
      // gettimeofday ( &timer, NULL );
      Status = tpuMemcpyH2DAsync(Dst, Src, Size, stream.stream());
      // gettimeofday ( &end, NULL );
      // std::cout << " tpuRtMemcpyS2DAsync getstream time : " << ( timer.tv_sec - timer1.tv_sec ) * 1000000UL + ( timer.tv_usec - timer1.tv_usec ) << "us\n";
      // std::cout << " sgMemcpyS2SAsync  time : " << ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec ) << "us\n";
    }
    TORCH_CHECK ( Status == tpuSuccess, "Failed to copy memory from host to TPU device #", Index, " size = ", Size, "bytes" );
  }

  void CopyDeviceToHost ( void * Dst, const void * Src, size_t Size, int Index, bool non_blocking )
  {
    tpuError_t Status;
    auto stream = c10_tpu::getDefaultTPUStream();
    tpudnnFlush(stream);
    tpuStreamSynchronize(stream.stream());
    if(!non_blocking) {
      Status = tpuMemcpyD2H(Dst, Src, Size);
    } else {
      // struct timeval timer1, timer, end;
      // gettimeofday ( &timer1, NULL );
      // gettimeofday ( &timer, NULL );
      Status = tpuMemcpyD2HAsync(Dst, Src, Size, stream.stream());
      // gettimeofday ( &end, NULL );
      // std::cout << " tpuRtMemcpyD2SAsync getstream time : " << ( timer.tv_sec - timer1.tv_sec ) * 1000000UL + ( timer.tv_usec - timer1.tv_usec ) << "us\n";
      // std::cout << " tpuRtMemcpyD2SAsync  time : " << ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec ) << "us\n";
    }
    TORCH_CHECK ( Status == tpuSuccess, "Failed to copy memory from TPU device #", Index, " to host size = ", Size, "bytes" );
  }

  void CopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, int Index, bool non_blocking )
  {
    auto stream = c10_tpu::getDefaultTPUStream();
    tpudnnTensor_t input = {.addr = tpudnnPhysToVirt(stream, (uint64_t)Src), .dim = 1, .dtype = TPUDNN_DTYPE_INT8};
    tpudnnTensor_t output = {.addr = tpudnnPhysToVirt(stream, (uint64_t)Dst), .dim = 1, .dtype = TPUDNN_DTYPE_INT8};
    input.shape[0] = output.shape[0] = Size;
    input.stride[0] = output.stride[0] = 1;
    auto Status = tpudnnStridedCopyAsync(stream, input, output);
    TORCH_CHECK (Status == TPUDNN_STATUS_SUCCESS, "TPU device #", Index, "D2D failed! Error Code : #", Status);
  }

  size_t getCurrentAllocSize(int Index)
  {
    return alloc_current_[Index];
  }
  size_t getPeakAllocSize(int Index)
  {
    return alloc_peak_[Index];
  }
  void resetPeakAllocSize(int Index)
  {
    std::lock_guard<std::mutex> lock(getMutex(Index));
    alloc_peak_[Index] = alloc_current_[Index];
  }

private:
  std::mutex& getMutex(int i) { return mutexes_[i]; }
  std::unordered_map<void *, MemInfo>& getMemInfo(int i) { return mem_info_[i]; }
  std::unordered_map<void *, MemToFree> &getFreeTBDs(int i) { return free_tbds_[i]; }

  std::vector<std::mutex> mutexes_;
  std::vector<std::unordered_map<void *, MemToFree>> free_tbds_;
  std::vector<std::unordered_map<void *, MemInfo>> mem_info_;

  std::vector<size_t> alloc_current_;
  std::vector<size_t> alloc_peak_;

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

void TPUDeleteInstance()
{
  if (auto stream = c10_tpu::getDefaultTPUStream()) {
    stream.synchronize();
    KernelManager::Instance()->UnloadKernelModule(stream.stream());
    tpuStreamDestroy(stream.stream());
  }
  TPUDeviceManager::DeleteInstance();
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
  auto Status = tpuGetDevice( &DevIndex );
  if (Status == tpuErrorNoDevice) {
    DevIndex = 0;
    TPUDeviceInitialize(DevIndex);
  } else
  {
    TORCH_CHECK (Status == tpuSuccess, " sgGetDevice failed! Error Code : #", Status);
  }
  if (!VISIBLE_DEVICE_NUM)
    return DevIndex;
  else
    return DevIndex % VISIBLE_DEVICE_NUM;
}

void TPUSetDeviceIndex ( int Index )
{
  auto Status = tpuSetDevice( Index );
  TORCH_CHECK (Status == tpuSuccess, " sgSetDevice failed! Error Code : #", Status);
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

void TPUEmptyCache ()
{
  TPUDeviceManager::GetInstance().EmptyCache( TPUGetDeviceIndex() );
}

void TPUEnableCache ()
{
  TPUDeviceManager::GetInstance().StartCache( TPUGetDeviceIndex() );
}

void TPUDisableCache ()
{
  TPUDeviceManager::GetInstance().StopCache( TPUGetDeviceIndex() );
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

tpuStream_t TPUGetDeviceResource ( void ) {
  auto stream = c10_tpu::getCurrentTPUStream();
  tpudnnFlush(stream);
  return stream.stream();
}

void TPUGetC2CRing(int world_size, int *chipMap)
{
  tpudnnStatus_t status = tpudnnGetC2CRing(world_size, chipMap);
  TORCH_CHECK (status == TPUDNN_STATUS_SUCCESS, " TPUGetC2CRing failed! Error Code : #", status);
}

void TPUSetupC2CTopology()
{
  auto status = tpuSetupTopology();
  TORCH_CHECK (status == tpuSuccess, " TPUSetupC2CTopology failed! Error Code : #", status);
}

void TPUGetTopology(std::vector<std::vector<int>> *topo) {
  int dev_cnt = 0;
  auto Status = tpuGetDeviceCount( &dev_cnt );
  TORCH_CHECK ( Status == tpuSuccess, "Failed to get TPU device count" );

  for (int i = 0; i < dev_cnt; ++i) {
      for (int j = 0; j < dev_cnt; ++j) {
          (*topo)[i][j] = -1;
      }
  }

  tpuTopology_t topology[dev_cnt][dev_cnt];
  Status = tpuGetTopology((tpuTopology_t **)topology);
  TORCH_CHECK (Status == tpuSuccess, " sgGetTopology failed! Error Code : #", Status);

  for (int i = 0; i < dev_cnt; i++) {
      for (int j = 0; j < dev_cnt; j++) {
          auto info = topology[i][j];
          if (info.parent_port != -1) {
              (*topo)[info.parent_dev][info.child_dev]=info.parent_port;
          }
      }
  }
}

int TPUCheckChipMap(int world_size, int *chipMap)
{
  tpudnnStatus_t status = tpudnnCheckChipMap(world_size, chipMap);
  return (int) status;
}

bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id) {
  int dev_cnt =  TPUGetDeviceCount();
  std::vector<std::vector<int>> topology(dev_cnt, std::vector<int>(dev_cnt, -1));
  TPUGetTopology(&topology);
  return topology[device_id][peer_device_id] != -1;
}

void TPUGetMemStats(std::unordered_map<std::string, int64_t> *stats, int index) {
  auto alloc_current = TPUDeviceManager::GetInstance().getCurrentAllocSize(index);
  auto alloc_peak = TPUDeviceManager::GetInstance().getPeakAllocSize(index);
  (*stats)["allocated_bytes.all.peak"] = alloc_peak;
  (*stats)["allocated_bytes.all.current"] = alloc_current;
}

void TPUGetMemInfo(size_t *free, size_t *total) {
  
  size_t totalMemSize = 0;
  size_t freeMemSize = 0;
  auto status = tpuGetAllMemory(&totalMemSize);
  TORCH_CHECK (status == tpuSuccess, " tpuGetAllMemory failed! Error Code : #", status);
  status = tpuGetFreeMemory(&freeMemSize);
  TORCH_CHECK (status == tpuSuccess, " tpuGetFreeMemory failed! Error Code : #", status);
  *total = totalMemSize;
  *free = freeMemSize;
}

void TPUResetPeakMemoryStats(int index)
{
  TPUDeviceManager::GetInstance().resetPeakAllocSize(index);
}

} // namespace tpu
