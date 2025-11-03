#pragma once

#include "tpuDNN.h"
#include "tpuDNNTensor.h"
#include <stddef.h>

extern "C"
{

enum scclDataType_t
{
    SCCL_DTYPE_FP32 = 0,
    SCCL_DTYPE_FP16 = 1,
    SCCL_DTYPE_INT8 = 2,
    SCCL_DTYPE_UINT8 = 3,
    SCCL_DTYPE_INT16 = 4,
    SCCL_DTYPE_UINT16 = 5,
    SCCL_DTYPE_INT32 = 6,
    SCCL_DTYPE_UINT32 = 7,
    SCCL_DTYPE_BF16 = 8,
    SCCL_DTYPE_INT4 = 9,
    SCCL_DTYPE_UINT4 = 10,
    SCCL_DTYPE_FP20 = 11,
    SCCL_DTYPE_FP8E5M2 = 12,
    SCCL_DTYPE_FP8E4M3 = 13,
    SCCL_DTYPE_INT64 = 14,
    SCCL_DTYPE_BOOL = 15,
    SCCL_DTYPE_UNKNOWN = -1,
};

enum scclReduceType_t {
    SCCL_REDUCE_MEAN = 0,
    SCCL_REDUCE_SUM  = 1,
    SCCL_REDUCE_MAX  = 2,
    SCCL_REDUCE_MIN  = 3,
    SCCL_REDUCE_PROD = 4,
    SCCL_REDUCE_L2   = 5,
    SCCL_REDUCE_L1   = 6,
};

typedef enum {
  scclSuccess = 0,
  scclKernelError = 1,
  scclInvalidArgument = 2,
} scclResult_t;

typedef void *scclComm_t;

typedef tpudnnHandle_t scclHandle_t;

#define SCCL_UNIQUE_ID_BYTES (128)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
typedef struct {
  char internal[SCCL_UNIQUE_ID_BYTES];
} scclUniqueId;

void *scclPhysToVirt(scclHandle_t handle, uint64_t addr);

scclResult_t scclSetupC2CTopology();

scclHandle_t scclCreateHandle(int deviceID);

void scclDestoryHandle(scclHandle_t handle);

scclResult_t scclGetUniqueId(scclHandle_t handle, scclUniqueId *uniqueId);

scclResult_t scclCommInitRank(scclComm_t *comm, int nRanks, scclUniqueId uniqueId,
                              int rank, const int *chipMap);
scclResult_t scclCommDestroy(scclComm_t comm);

scclResult_t scclSend(const void *send_buff, uint64_t send_count,
                           scclDataType_t dtype, int dst_rank,
                           scclComm_t comm, scclHandle_t handle);
scclResult_t scclRecv(const void *recv_buff, uint64_t recv_count,
                           scclDataType_t dtype, int src_rank,
                           scclComm_t comm, scclHandle_t handle);
scclResult_t scclAllGather(const void *sendBuff, void *recvBuff,
                           uint64_t send_count, scclDataType_t dtype,
                           scclComm_t comm, scclHandle_t handle);
scclResult_t scclBroadcast(void *buff, uint64_t count, scclDataType_t dtype,
                           int root, scclComm_t comm, scclHandle_t handle);
scclResult_t scclAllReduce(const void *sendBuff, void *recvBuff, uint64_t count,
                           scclDataType_t dtype, scclReduceType_t op,
                           scclComm_t comm, scclHandle_t handle);
scclResult_t scclReduce(const void *sendBuff, void *recvBuff, uint64_t count,
                        scclDataType_t dtype, scclReduceType_t op, int root,
                        scclComm_t comm, scclHandle_t handle);
scclResult_t scclGather(const void *sendBuff, void *recvBuff, uint64_t sendcount,
                        scclDataType_t dtype, int root, scclComm_t comm,
                        scclHandle_t handle);
scclResult_t scclScatter(const void *sendBuff, void *recvBuff,
                         uint64_t recv_count, scclDataType_t dtype, int root,
                         scclComm_t comm, scclHandle_t handle);
scclResult_t scclAllToAll(const void *sendBuff, void *recvBuff,
                          uint64_t recv_count, scclDataType_t dtype,
                          scclComm_t comm, scclHandle_t handle);

} // extern "C"
