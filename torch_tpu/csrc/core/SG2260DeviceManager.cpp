#include <vector>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <c10/util/Logging.h>
#include <c10/core/DeviceType.h>
#include "TPUDeviceManager.h"

#ifdef BACKEND_SG2260
#include "TPUStream.h"
#include <sgdnn_api.h>
#include <tpuv7_rt.h>
namespace tpu
{
using Devptr = unsigned char *;

class TPUDeviceManager
{
public:
  TPUDeviceManager() : init_flag_(false) {}

  ~TPUDeviceManager() {
    auto stream_ = c10_tpu::getCurrentTPUStream();
    sgdnnDeinitialize(stream_);
    tpuRtStreamSynchronize(stream_);
    tpuRtStreamDestroy(stream_);
    if (instance_) delete instance_;
  }

  TPUMgrStatus Initialized() { return init_flag_ ? INIT_ALREADY : NOT_INIT;}

  TPUMgrStatus initialize()
  {
    if (init_flag_) return INIT_SUCCESS;
    int DeviceCount = 0;
    tpuRtStatus_t Status = tpuRtGetDeviceCount ( &DeviceCount );
    if (DeviceCount == 0) {
      std::cout << "Device Count:" << DeviceCount << "\n";
      DeviceCount = 1;
    }
    // TODO multi-device
    TORCH_CHECK ( DeviceCount == 1 );
    tpuRtInit();
    tpuRtSetDevice(0);

    TORCH_CHECK ( Status == tpuRtSuccess, "Failed to get TPU device count" );
    if ( DeviceCount > 0 )
    {
      Mutexes_ = std::vector<std::mutex> ( DeviceCount );
      AddrMemMaps_ = std::vector<std::unordered_set<Devptr>> ( DeviceCount );
      devices_init_ = std::vector<std::atomic<bool>> (DeviceCount);
      devices_init_[0] = 1; // tpuRtDeviceInit will default set idx = 0 device. that not a good idea.
    }
    SOPHON_LOG("TPU Device Manager init successfully");
    init_flag_ = true;
    return INIT_SUCCESS;
  }

  TPUMgrStatus InitDevice(int Index ){
    tpuRtStatus_t Status = tpuRtSetDevice( Index );
    TORCH_CHECK (Status == tpuRtSuccess, " sgSetDevice failed! Error Code : #", Status);
    return INIT_SUCCESS;
  }

  static TPUDeviceManager* GetInstance()
  {
    if (instance_ == nullptr){
      instance_ = new TPUDeviceManager();
    }
    instance_->initialize();
    return instance_;
  }

  int GetDeviceCount() const
  {
    return ( int ) AddrMemMaps_.size();
  }

  void * Alloc ( size_t Size, int Index )
  {
    Mutexes_[Index].lock();
    if ( Size == 0 )
    {
      Mutexes_[Index].unlock();
      return nullptr;
    }
    Devptr dev_ptr;
    tpuRtStatus_t Status = tpuRtMalloc((void **)(&dev_ptr), Size);
    TORCH_CHECK ( Status == tpuRtSuccess, "Failed to allocate memory on TPU device #", Index, " size = ", Size, "bytes" );
    AddrMemMaps_[Index].insert ( dev_ptr );

    Mutexes_[Index].unlock();
    return ( void * ) UnifiedAddr((unsigned long long) dev_ptr, Index);
  }

  void Free ( void * Ptr, int Index )
  {
    Mutexes_[Index].lock();
    if ( Ptr == nullptr )
    {
      Mutexes_[Index].unlock();
      return;
    }
    auto Iter = AddrMemMaps_[Index].find ( ( Devptr ) Ptr );
    TORCH_CHECK ( Iter != AddrMemMaps_[Index].end(), "Memory of address = ", Ptr, " is not found" );
    AddrMemMaps_[Index].erase ( (Devptr) Ptr );
    tpuRtFree((Devptr) Ptr);
    Mutexes_[Index].unlock();
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
  std::mutex& get_handle_mutex(int i) { return Mutexes_[i]; }
  std::unordered_set<Devptr>& getAddrMems(int i) { return AddrMemMaps_[i]; }

  std::vector<std::mutex> Mutexes_;
  std::vector<std::unordered_set<Devptr>> AddrMemMaps_;
  std::atomic<bool> init_flag_;
  std::vector<std::atomic<bool>> devices_init_;

  static TPUDeviceManager* instance_;
};

TPUDeviceManager* TPUDeviceManager::instance_ = nullptr;

TPUDeviceManager* TPUGetInstance(){
  return TPUDeviceManager::GetInstance();
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
  return TPUDeviceManager::GetInstance()->InitDevice(Index);
}

int TPUGetDeviceCount ( void )
{
  return TPUDeviceManager::GetInstance()->GetDeviceCount();
}

int TPUGetDeviceIndex ( void )
{
  int DevIndex;
  tpuRtStatus_t Status = tpuRtGetDevice( &DevIndex );
  if (Status = tpuRtErrorNoDevice) {
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
  return TPUDeviceManager::GetInstance()->Alloc ( Size, TPUGetDeviceIndex() );
}

void * TPUAlloc ( size_t Size, int Index )
{
  return TPUDeviceManager::GetInstance()->Alloc ( Size, Index );
}

void TPUFree ( void * Ptr )
{
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Ptr);
  unsigned long long data_ptr = GetAddrByUnifiedAddr((unsigned long long)Ptr);
  TPUDeviceManager::GetInstance()->Free ( (void *)data_ptr, dev_index );
}


void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size,  bool non_blocking)
{
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Dst);
  unsigned long long dst_ptr = GetAddrByUnifiedAddr((unsigned long long)Dst);
  TPUDeviceManager::GetInstance()->CopyHostToDevice ( (void*)dst_ptr, Src, Size, dev_index, non_blocking );
}

void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size, bool non_blocking )
{
  unsigned long long dev_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Src);
  unsigned long long src_ptr = GetAddrByUnifiedAddr((unsigned long long)Src);
  TPUDeviceManager::GetInstance()->CopyDeviceToHost ( Dst, (void*)src_ptr, Size, dev_index, non_blocking );
}

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking )
{
  unsigned long long src_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Src);
  unsigned long long src_ptr = GetAddrByUnifiedAddr((unsigned long long)Src);
  unsigned long long dst_index = GetDeviceIndexByUnifiedAddr((unsigned long long)Dst);
  unsigned long long dst_ptr = GetAddrByUnifiedAddr((unsigned long long)Dst);
  TORCH_CHECK ( dst_index == src_index, "D2D copy must in same device");
  TPUDeviceManager::GetInstance()->CopyDeviceToDevice ( (void*)dst_ptr, (void*)src_ptr, Size, dst_index, non_blocking );
}

tpuRtStream_t TPUGetDeviceResource ( void ) {
  return c10_tpu::getCurrentTPUStream();
}

} // namespace tpu

#endif //BACKEND_SG2260