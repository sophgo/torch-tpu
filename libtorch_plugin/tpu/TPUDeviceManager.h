#pragma once

#include <bmlib_runtime.h>

namespace tpu
{
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
 * Get the current TPU device handle.
 */
bm_handle_t TPUGetDeviceHandle ( void );

/**
 * Alloc memory on the TPU current device.
 */
void * TPUAlloc ( size_t Size );

/**
 * Free memory allocted on arbitrary TPU device.
 */
void TPUFree ( void * Ptr );

/**
 * Copy data from host to TPU device.
 */
void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size );

/**
 * Copy data from TPU device to host.
 */
void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size );

bool TPUPtrIsInCurrentDevice ( const void * Ptr );

void * TPUGetAddrInDevice ( const void * Ptr );

} // namespace tpu
