/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __TPUV7_H__
#define __TPUV7_H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MD5SUM_LEN 16
#define LIB_MAX_NAME_LEN 64
#define FUNC_MAX_NAME_LEN 64

typedef enum {
	tpuRtSuccess = 0,
	tpuRtDevnotready = 1, /* Device not ready yet */
	tpuRtErrFailure = 2,     /* General failure */
	tpuRtErrTimeout = 3,     /* Timeout */
	tpuRtErrParam = 4,       /* Parameters invalid */
	tpuRtErrNomem = 5,       /* Not enough memory */
	tpuRtErrData = 6,        /* Data error */
	tpuRtErrBusy = 7,        /* Busy */
	tpuRtErrNoFeature = 8,   /* Not supported yet */
	tpuRtErrNotSupported = 9,
	tpuRtErrorInitializationError,
	tpuRtErrorNoDevice,
	tpuRtErrorNoStream,
	tpuRtErrorInvalidValue,
	tpuRtErrorInvalidResourceHandle,
	tpuRtErrorMemoryAllocation
} tpuRtStatus_t;

typedef struct tpuRtModule *tpuRtKernelModule_t;
typedef struct tpuRtStream *tpuRtStream_t;
typedef struct tpuRtEvent *tpuRtEvent_t;
typedef struct timeval tpuRtTimeRecord;
typedef int (*pTpuRtStreamCallback)(void *);

void sgTimeRecord(tpuRtTimeRecord *time);
uint32_t sgTimeCalculate(tpuRtTimeRecord time_start, tpuRtTimeRecord time_end);

tpuRtStatus_t tpuRtInit(void);
tpuRtStatus_t tpuRtGetDeviceCount(int *count);
tpuRtStatus_t tpuRtGetDevice(int *device);
tpuRtStatus_t tpuRtSetDevice(int device);
tpuRtStatus_t tpuRtDeviceSynchronize(void);
tpuRtStatus_t tpuRtMalloc(void **devPtr, unsigned long long size, int parallel_num);
tpuRtStatus_t tpuRtFree(void **devPtr, int free_num);
tpuRtStatus_t tpuRtMallocHost(void **ptr, unsigned long long size);
tpuRtStatus_t tpuRtFreeHost(void *ptr);
tpuRtStatus_t tpuRtMemset(void *devPtr, int value, unsigned long long size);
tpuRtStatus_t tpuRtMemcpyS2D(void *devPtr, const void *hostPtr, unsigned long long size);
tpuRtStatus_t tpuRtMemcpyD2S(void *hostPtr, const void *devPtr, unsigned long long size);
tpuRtStatus_t tpuRtMemcpyD2D(void *dstDevPtr, const void *srcDevPtr, unsigned long long size);
tpuRtStatus_t tpuRtMemcpyP2P(void *dstDevPtr, int dstDevice, const void *srcDevPtr, int srcDevice,
			  unsigned long long size);
tpuRtStatus_t tpuRtMemsetAsync(void *devPtr, int value, unsigned long long size, tpuRtStream_t stream);
tpuRtStatus_t tpuRtMemcpyS2DAsync(void *devPtr, const void *hostPtr, unsigned long long size, tpuRtStream_t stream);
tpuRtStatus_t tpuRtMemcpyD2SAsync(void *hostPtr, const void *devPtr, unsigned long long size, tpuRtStream_t stream);
tpuRtStatus_t tpuRtMemcpyD2DAsync(void *dstDevPtr, const void *srcDevPtr, unsigned long long size,
				  tpuRtStream_t stream);
tpuRtStatus_t tpuRtMemcpyP2PAsync(void *dstDevPtr, int dstDevice, const void *srcDevPtr,
				  int srcDevice, unsigned long long size, tpuRtStream_t stream);
tpuRtStatus_t tpuRtStreamCreate(tpuRtStream_t *pStream);
tpuRtStatus_t tpuRtStreamDestroy(tpuRtStream_t stream);
tpuRtStatus_t tpuRtStreamSynchronize(tpuRtStream_t stream);
tpuRtStatus_t tpuRtStreamAddCallback(tpuRtStream_t stream, pTpuRtStreamCallback callback, void *userData);
tpuRtStatus_t tpuRtStreamWaitEvent(tpuRtStream_t stream, tpuRtEvent_t event);
tpuRtStatus_t tpuRtEventCreate(tpuRtEvent_t *pEvent);
// tpuRtStatus_t sgEventDestroy(tpuRtEvent_t event);
tpuRtStatus_t tpuRtEventRecord(tpuRtEvent_t event, tpuRtStream_t stream);
tpuRtStatus_t tpuRtEventSynchronize(tpuRtEvent_t event);
tpuRtStatus_t tpuRtEventElapsedTime(float *ms, tpuRtEvent_t start, tpuRtEvent_t end);

tpuRtKernelModule_t tpuRtKernelLoadModuleFile(const char *module_file, tpuRtStream_t stream);
tpuRtKernelModule_t tpuRtKernelLoadModule(const char *data, size_t length, tpuRtStream_t stream);
tpuRtStatus_t tpuRtKernelLaunch(tpuRtKernelModule_t module, const char *func_name, void *args, uint32_t size,
			      uint64_t group_num, uint64_t block_num, tpuRtStream_t stream);
tpuRtStatus_t tpuRtKernelLaunchAsync(tpuRtKernelModule_t module, const char *func_name, void *args,
				   uint32_t size, uint64_t group_num, uint64_t block_num, tpuRtStream_t stream);
tpuRtStatus_t tpuRtKernelUnloadModule(tpuRtKernelModule_t p_module, tpuRtStream_t stream);

uint64_t tpuRtGetClusterId(void);
tpuRtStatus_t tpuRtKernelLaunchCDMA(tpuRtKernelModule_t module, const char *func_name,
				void *args, uint32_t size, uint64_t block_num, tpuRtStream_t stream,
				int cdma_only, uint64_t unique_id, int rank_id, int rank_num);
tpuRtStatus_t tpuRtKernelLaunchCDMAAsync(tpuRtKernelModule_t module, const char *func_name,
				void *args, uint32_t size, uint64_t block_num, tpuRtStream_t stream,
				int cdma_only, uint64_t unique_id, int rank_id, int rank_num);
#ifdef __cplusplus
}
#endif
#endif
