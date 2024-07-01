#pragma once

#include "sophon_defines_2260.h"
#include <stddef.h>

namespace sophon {
typedef enum {
  scclSuccess = 0,
  scclKernelError = 1,
  scclInvalidArgument = 2,
} scclResult_t;

typedef void *scclComm_t;

#define SCCL_UNIQUE_ID_BYTES (128)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
typedef struct {
  char internal[SCCL_UNIQUE_ID_BYTES];
} scclUniqueId;

scclResult_t scclCommInitRank(scclComm_t *comm, int nranks, scclUniqueId commId,
                              int rank, const int *chip_map);
scclResult_t scclCommDestroy(scclComm_t comm);

scclResult_t scclAllGather(const void *send_buff, void *recv_buff,
                           size_t send_count, tpudnnDataType_t dtype,
                           scclComm_t comm, tpudnnHandle_t handle);
scclResult_t scclBroadcast(void *buff, size_t count, tpudnnDataType_t dtype,
                           int root, scclComm_t comm, tpudnnHandle_t handle);
scclResult_t scclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           tpudnnDataType_t dtype, tpudnnReduceType_t op,
                           scclComm_t comm, tpudnnHandle_t handle);
scclResult_t scclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        tpudnnDataType_t dtype, tpudnnReduceType_t op, int root,
                        scclComm_t comm, tpudnnHandle_t handle);
scclResult_t scclGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                        tpudnnDataType_t dtype, int root, scclComm_t comm,
                        tpudnnHandle_t handle);
scclResult_t scclScatter(const void *sendbuff, void *recvbuff,
                         size_t recv_count, tpudnnDataType_t dtype, int root,
                         scclComm_t comm, tpudnnHandle_t handle);
scclResult_t scclAllToAll(const void *sendbuff, void *recvbuff,
                          size_t recv_count, tpudnnDataType_t dtype,
                          scclComm_t comm, tpudnnHandle_t handle);
} // namespace sophon