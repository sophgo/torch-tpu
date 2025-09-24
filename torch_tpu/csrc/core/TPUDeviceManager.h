#pragma once
#include <torch/torch.h>

#include "torch_tpu/csrc/common/config.h"
#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUAddrHelper.h"
#include <tpu_runtime_api.h>
namespace tpu
{


enum TPUMgrStatus{
    NOT_INIT     = -1,
    INIT_SUCCESS = 0,
    INIT_ALREADY,
    INIT_FAILED,
    DESTORY_SUCCESS,
};

class TPUDeviceManager;
TPUDeviceManager* TPUGetInstance();
void TPUDeleteInstance();

TPUMgrStatus InitTPUMgr();
TPUMgrStatus DestoryTpuMgr();
TPUMgrStatus IsTPUMgrInited();
TPUMgrStatus TPUDeviceInitialize( int Index);

/**
 * Get the count of TPU devices.
 */
int TPUGetDeviceCount ( void );

/**
 * Get the current TPU device index.
 */
int TPUGetDeviceIndex ( void );

/**
 * Set the current TPU device index.
 */
void TPUSetDeviceIndex ( int Index );

/**
 * Alloc memory on the TPU current device.
 */
void * TPUAlloc ( size_t Size );
void * TPUAlloc ( size_t Size, int Index );

/**
 * Free memory allocted on arbitrary TPU device.
 */
void TPUFree ( void * Ptr );

/**
 * Free cached Memory
 */
void TPUEmptyCache ();

/**
 * Copy data from host to TPU device.
 */
void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking = false );

/**
 * Copy data from TPU device to host.
 */
void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size, bool non_blocking = false);

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking = true);

tpuStream_t TPUGetDeviceResource ( void );

/**
 * multi-chip Topology utils
 */
void TPUSetupC2CTopology();
void TPUGetTopology(std::vector<std::vector<int>> *topo);
void TPUGetC2CRing(int world_size, int *chipMap);
int  TPUCheckChipMap(int world_size, int *chipMap);

/**
 * Memory stats related.
 */
void TPUGetMemStats(std::unordered_map<std::string, int64_t> *stats, int index);
void TPUGetMemInfo(size_t* free, size_t* total);
void TPUResetPeakMemoryStats(int index);

bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id);

} // namespace tpu
