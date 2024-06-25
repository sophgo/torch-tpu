#ifndef SGDNN_RUNTIME_H
#define SGDNN_RUNTIME_H

#if defined BACKEND_SG2260
#include "tpuv7_rt.h"
#else
#include "bmlib_runtime.h"
#endif
#include <stdio.h>

#ifndef NO_USE
#define NO_USE 0
#endif

#if defined BACKEND_SG2260
typedef tpuRtStream_t tpu_resource_t;
typedef tpuRtStatus_t tpu_status_t;
typedef void* tpu_device_mem_t;
#define SG_SUCCESS tpuRtSuccess
#else
typedef bm_handle_t tpu_resource_t;
typedef bm_status_t tpu_status_t;
typedef bm_device_mem_t tpu_device_mem_t;
#define SG_SUCCESS BM_SUCCESS
#endif

#define SGDNN_CHECK(expression) \
do \
{ \
  if ( !( expression ) ) \
  { \
    printf ( "%s:%d:%s: %s failed\n", __FILE__, __LINE__, __func__, #expression ); \
    throw; \
  } \
} \
while ( false )

#define SAFE_CALL(cmd) \
do \
{ \
  if ( !( ( cmd ) == SG_SUCCESS ) ) \
  { \
    printf ( "%s:%d:%s: %s failed\n", __FILE__, __LINE__, __func__, #cmd ); \
    throw; \
  } \
} \
while ( false )

static inline tpu_status_t sgdnnGetDeviceCount(int *count) {
#if defined BACKEND_1684X
  return bm_dev_getcount(count);
#elif defined BACKEND_SG2260
  return tpuRtGetDeviceCount(count);
#else
  SGDNN_CHECK ( false );
#endif
}

static inline tpu_status_t sgdnnMallocDeviceByte(tpu_resource_t tpu_resource,
                                                tpu_device_mem_t *pmem, unsigned long long size) {
#if defined BACKEND_1684X
  return bm_malloc_device_byte(tpu_resource, pmem, size);
#elif defined BACKEND_SG2260
  return tpuRtMalloc(pmem, size, NO_USE);
#else
  SGDNN_CHECK ( false );
#endif
}

static inline unsigned long long sgdnnGetDeviceAddr(tpu_device_mem_t device_mem) {
#if defined BACKEND_1684X
  return bm_mem_get_device_addr(device_mem);
#elif defined BACKEND_SG2260
  return (unsigned long long)device_mem;
#else
  SGDNN_CHECK ( false );
#endif
}

static inline void sgdnnFreeDevice(tpu_resource_t tpu_resource, tpu_device_mem_t device_mem) {
#if defined BACKEND_1684X
  bm_free_device(tpu_resource, device_mem);
#elif defined BACKEND_SG2260
  tpuRtStreamSynchronize(tpu_resource);
  tpuRtFree(&device_mem, NO_USE);
#else
  SGDNN_CHECK ( false );
#endif
}

static inline tpu_device_mem_t sgdnnMemFromDevice(unsigned long long device_addr, unsigned long long len) {
#if defined BACKEND_1684X
  return bm_mem_from_device(device_addr, len);
#elif defined BACKEND_SG2260
  return (void *)device_addr;
#else
  SGDNN_CHECK ( false );
#endif
}

static inline tpu_status_t sgdnnMemcpyD2S(tpu_resource_t tpu_resource, void *dst, tpu_device_mem_t src, unsigned long long size) {
#if defined BACKEND_1684X
  return bm_memcpy_d2s(tpu_resource, dst, src);
#elif defined BACKEND_SG2260
  return tpuRtMemcpyD2SAsync(dst, src, size, tpu_resource);
#else
  SGDNN_CHECK ( false );
#endif
}

static inline tpu_status_t sgdnnMemcpyS2D(tpu_resource_t tpu_resource, tpu_device_mem_t dst, void *src, unsigned long long size) {
#if defined BACKEND_1684X
  return bm_memcpy_s2d(tpu_resource, dst, src);
#elif defined BACKEND_SG2260
  return tpuRtMemcpyS2DAsync(dst, src, size, tpu_resource);
#else
  SGDNN_CHECK ( false );
#endif
}

static inline tpu_status_t sgdnnMemcpyD2D(tpu_resource_t tpu_resource, tpu_device_mem_t dst,
                                          unsigned long long dst_offset, tpu_device_mem_t src,
                                          unsigned long long src_offset, unsigned long long size) {
#if defined BACKEND_1684X
  return bm_memcpy_d2d_byte(tpu_resource, dst, dst_offset, src, src_offset, size);
#elif defined BACKEND_SG2260
  SGDNN_CHECK(dst_offset == 0 && src_offset == 0);
  return tpuRtMemcpyD2DAsync(dst, src, size, tpu_resource);
#else
  SGDNN_CHECK ( false );
#endif
}

#endif //end def SGDNN_RUNTIME_H
