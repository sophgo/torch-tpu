#pragma once
#include <torch/torch.h>

#include "torch_tpu/csrc/common/config.h"
#include "torch_tpu/csrc/core/TPULog.h"
#ifdef BACKEND_1684X
#include <bmlib_runtime.h>
#endif

#define TPU_DEVICE_INDEX_BITS 6
#define TPU_GLOBAL_ADDR_BITS (64 - TPU_DEVICE_INDEX_BITS)

namespace tpu
{
static inline unsigned long long UnifiedAddr( unsigned long long Addr, int Index)
{
  TORCH_CHECK ( Addr < ( 1UL << TPU_GLOBAL_ADDR_BITS ) );
  return ( ( ( unsigned long long ) Index ) << TPU_GLOBAL_ADDR_BITS ) | Addr;
}

static inline unsigned long long GetDeviceIndexByUnifiedAddr ( unsigned long long Addr )
{
  return Addr >> TPU_GLOBAL_ADDR_BITS;
}

static inline unsigned long long GetAddrByUnifiedAddr ( unsigned long long Addr )
{
  return ( Addr << TPU_DEVICE_INDEX_BITS ) >> TPU_DEVICE_INDEX_BITS;
}


enum TPUMgrStatus{
    NOT_INIT     = -1,
    INIT_SUCCESS = 0,
    INIT_ALREADY,
    INIT_FAILED,
    DESTORY_SUCCESS,
};

class TPUDeviceManager;
TPUDeviceManager* TPUGetInstance();

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
 * Copy data from host to TPU device.
 */
void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking = false );

/**
 * Copy data from TPU device to host.
 */
void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size, bool non_blocking = false);

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking = false);

#ifdef BACKEND_1684X
/**
 * Get the current TPU device handle.
 */
bm_handle_t TPUGetDeviceHandle ( void );
#endif

} // namespace tpu