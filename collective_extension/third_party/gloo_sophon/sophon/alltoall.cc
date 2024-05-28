/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/alltoall.h"

#include <cstring>

#include "sophon/common/logging.h"
#include "sophon/types.h"

namespace sophon {

void alltoall(AlltoallOptions& opts) {
  const auto& context = opts.context;
  transport::UnboundBuffer* in = opts.in.get();
  transport::UnboundBuffer* out = opts.out.get();
  const auto slot = Slot::build(kAlltoallSlotPrefix, opts.tag);

  // Sanity checks.
  // Number of elements should be evenly split in input and output buffers.
  SOPHON_ENFORCE(opts.elementSize > 0);
  SOPHON_ENFORCE(in != nullptr);
  SOPHON_ENFORCE(out != nullptr);
  SOPHON_ENFORCE(in->size % context->size == 0);
  SOPHON_ENFORCE(in->size == out->size);

  size_t chunkSize = in->size / context->size;
  int myRank = context->rank;
  int worldSize = context->size;

  // Local copy.
  memcpy(
      static_cast<char*>(out->ptr) + myRank * chunkSize,
      static_cast<char*>(in->ptr) + myRank * chunkSize,
      chunkSize);

  // Remote copy.
  for (int i = 1; i < worldSize; i++) {
    int sendRank = (myRank + i) % worldSize;
    int recvRank = (myRank + worldSize - i) % worldSize;
    in->send(sendRank, slot, sendRank * chunkSize, chunkSize);
    out->recv(recvRank, slot, recvRank * chunkSize, chunkSize);
  }

  for (int i = 1; i < worldSize; i++) {
    in->waitSend(opts.timeout);
    out->waitRecv(opts.timeout);
  }
}

void alltoall2260( AlltoallOptions &opts) {
  // call tpudnnC2CAllToAll
  sccl_args_t sccl_args;
  sccl_args.nranks = opts.context->size;
  sccl_args.rank = opts.context->rank;
  for (int i = 0; i < sccl_args.nranks; i++) {
    sccl_args.chip_map[i] = i;
  }
  tpudnnStatus_t ret =
      tpudnnC2CAllToAll(opts.handle_, tpudnnPhysToVirt(opts.handle_, (uint64_t)opts.send_buff_), opts.elements, opts.sg_type_,
                                       tpudnnPhysToVirt(opts.handle_, (uint64_t)opts.recv_buff_), opts.elements, opts.sg_type_, sccl_args);
  return;
}

} // namespace sophon
