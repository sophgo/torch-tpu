#include "sccl.h"
#include <string.h>

namespace sophon {

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

} // namespace sophon