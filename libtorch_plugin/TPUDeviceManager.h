#pragma once

#include <bmlib_runtime.h>

namespace c10
{
namespace tpu
{
int TPUGetDeviceCount ( void );
int TPUGetDeviceIndex ( void );
void TPUSetDeviceIndex ( int Index );
bm_handle_t TPUGetDeviceHandle ( void );
void * TPUAlloc ( size_t Size );
void TPUFree ( void * Ptr );
} // namespace tpu
} // namespace c10
