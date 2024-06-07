#pragma once

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

#define MAX_CHIP_NUM (16)
typedef struct {
  int rank;
  int nranks;
  int chip_map[MAX_CHIP_NUM];
  scclUniqueId unique_id;
} scclComm;

} // namespace sophon