#pragma once
#include <torch/torch.h>

#include "torch_tpu/csrc/common/config.h"
#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUAddrHelper.h"
#include "sgdnn_runtime.h"

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

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking = true);

#if defined BACKEND_1684X
/**
 * Get the current TPU device handle.
 */
bm_handle_t TPUGetDeviceResource ( void );
#elif defined(BACKEND_SG2260)
tpuRtStream_t TPUGetDeviceResource ( void );
#endif

} // namespace tpu