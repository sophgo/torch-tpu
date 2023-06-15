#pragma once

#include <bmlib_runtime.h>
#include "common/config.h"

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
void * TPUAlloc ( size_t Size, int Index );

/**
 * Free memory allocted on arbitrary TPU device.
 */
void TPUFree ( void * Ptr );
void TPUFree ( void * Ptr, int Index );

/**
 * Copy data from host to TPU device.
 */
void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size );

/**
 * Copy data from TPU device to host.
 */
void TPUCopyDeviceToHost ( void * Dst, const void * Src, size_t Size );

void TPUCopyDeviceToDevice ( void * Dst, const void * Src, size_t Size );

} // namespace tpu
