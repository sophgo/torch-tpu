#include "sccl.h"
#include <string.h>

namespace sophon {

#define MAX_CHIP_NUM (16)
typedef struct {
  int rank;
  int nranks;
  int chip_map[MAX_CHIP_NUM];
  scclUniqueId unique_id;
} scclComm;

scclResult_t scclCommInitRank(scclComm_t *comm, int nranks, scclUniqueId commId,
                              int rank, const int *chip_map) {
  if (nranks <= 0 || rank < 0 || rank > nranks) {
    return scclInvalidArgument;
  }

  auto pcomm = new scclComm;
  pcomm->nranks = nranks;
  pcomm->rank = rank;
  pcomm->unique_id = commId;
  if (chip_map == nullptr) {
    for (int i = 0; i < nranks; ++i) {
      pcomm->chip_map[i] = i;
    }
  } else {
    memcpy(pcomm->chip_map, chip_map, nranks * sizeof(int));
  }
  *comm = pcomm;
  return scclSuccess;
}
scclResult_t scclCommDestroy(scclComm_t comm) {
  if (comm == nullptr) {
    return scclSuccess;
  }

  auto pcomm = static_cast<scclComm *>(comm);
  if (pcomm->nranks <= 0 || pcomm->rank < 0 || pcomm->rank > pcomm->nranks) {
    return scclInvalidArgument;
  }

  delete static_cast<scclComm *>(pcomm);
  return scclSuccess;
}

static inline void ConvertToTpuDNNArgs(const scclComm_t comm,
                                       sccl_args_t *args) {
  memset(args, 0x0, sizeof(sccl_args_t));
  const scclComm *pcomm = static_cast<scclComm *>(comm);
  args->nranks = pcomm->nranks;
  args->rank = pcomm->rank;
  memcpy(args->chip_map, pcomm->chip_map, sizeof(int) * pcomm->nranks);
}

scclResult_t scclAllGather(const void *send_buff, void *recv_buff,
                           size_t send_count, sg_data_type_t dtype,
                           scclComm_t comm, tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  tpudnnStatus_t ret = tpudnnC2CAllGather(
      handle, tpudnnPhysToVirt(handle, (uint64_t)send_buff), send_count,
      tpudnnPhysToVirt(handle, (uint64_t)recv_buff), send_count,
      reinterpret_cast<const char *>(&pcomm->unique_id), dtype, args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}

scclResult_t scclBroadcast(void *buff, size_t count, sg_data_type_t dtype,
                           int root, scclComm_t comm, tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  auto ret = tpudnnC2CBroadcast(
      handle, tpudnnPhysToVirt(handle, (uint64_t)buff), count, dtype, root,
      reinterpret_cast<const char *>(&pcomm->unique_id), args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}

scclResult_t scclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           sg_data_type_t dtype, sg_reduce_method_t op,
                           scclComm_t comm, tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  auto ret = tpudnnC2CAllReduce(
      handle, tpudnnPhysToVirt(handle, (uint64_t)sendbuff),
      tpudnnPhysToVirt(handle, (uint64_t)recvbuff), count, dtype, op,
      reinterpret_cast<const char *>(&pcomm->unique_id), args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}

scclResult_t scclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        sg_data_type_t dtype, sg_reduce_method_t op, int root,
                        scclComm_t comm, tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  auto ret = tpudnnC2CReduce(
      handle, tpudnnPhysToVirt(handle, (uint64_t)sendbuff),
      tpudnnPhysToVirt(handle, (uint64_t)recvbuff), count, dtype, op, root,
      reinterpret_cast<const char *>(&pcomm->unique_id), args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}

scclResult_t scclGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                        sg_data_type_t dtype, int root, scclComm_t comm,
                        tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  auto ret = tpudnnC2CGather(
      handle, tpudnnPhysToVirt(handle, (uint64_t)sendbuff), sendcount,
      tpudnnPhysToVirt(handle, (uint64_t)recvbuff), sendcount * args.nranks,
      dtype, root, reinterpret_cast<const char *>(&pcomm->unique_id), args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}

scclResult_t scclScatter(const void *sendbuff, void *recvbuff,
                         size_t recv_count, sg_data_type_t dtype, int root,
                         scclComm_t comm, tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  auto ret = tpudnnC2CScatter(
      handle, tpudnnPhysToVirt(handle, (uint64_t)sendbuff),
      recv_count / args.nranks, dtype,
      tpudnnPhysToVirt(handle, (uint64_t)recvbuff), recv_count, dtype, root,
      reinterpret_cast<const char *>(&pcomm->unique_id), args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}

scclResult_t scclAllToAll(const void *sendbuff, void *recvbuff,
                          size_t recv_count, sg_data_type_t dtype,
                          scclComm_t comm, tpudnnHandle_t handle) {
  scclComm *pcomm = static_cast<scclComm *>(comm);
  sccl_args_t args = {0};
  ConvertToTpuDNNArgs(comm, &args);
  auto ret = tpudnnC2CAllToAll(
      handle, tpudnnPhysToVirt(handle, (uint64_t)sendbuff), recv_count, dtype,
      tpudnnPhysToVirt(handle, (uint64_t)recvbuff), recv_count, dtype,
      reinterpret_cast<const char *>(&pcomm->unique_id), args);
  return ret == TPUDNN_STATUS_SUCCESS ? scclSuccess : scclKernelError;
}
} // namespace sophon