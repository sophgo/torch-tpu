/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/gather.h"

#include <cstring>

#include "sophon/common/logging.h"
#include "sophon/types.h"

namespace sophon {

void gather(GatherOptions& opts) {
  const auto& context = opts.context;
  transport::UnboundBuffer* in = opts.in.get();
  transport::UnboundBuffer* out = opts.out.get();
  const auto slot = Slot::build(kGatherSlotPrefix, opts.tag);

  // Sanity checks
  SOPHON_ENFORCE(opts.elementSize > 0);
  SOPHON_ENFORCE(in != nullptr);

  if (context->rank == opts.root) {
    const size_t chunkSize = in->size;

    // Ensure the output buffer has the right size.
    SOPHON_ENFORCE(out != nullptr);
    SOPHON_ENFORCE(in->size * context->size == out->size);

    // Post receive operations from peers into out buffer
    for (size_t i = 0; i < context->size; i++) {
      if (i == context->rank) {
        continue;
      }
      out->recv(i, slot, i * chunkSize, chunkSize);
    }

    // Copy local input to output
    memcpy(static_cast<char*>(out->ptr) + (context->rank * chunkSize), in->ptr,
           chunkSize);

    // Wait for receive operations to complete
    for (size_t i = 0; i < context->size; i++) {
      if (i == context->rank) {
        continue;
      }
      out->waitRecv(opts.timeout);
    }
  } else {
    in->send(opts.root, slot);
    in->waitSend(opts.timeout);
  }
}

void gather2260(GatherOptions &opts) {
  // call tpudnnC2CGather
  sccl_args_t sccl_args;
  sccl_args.nranks = opts.context->size;
  sccl_args.rank = opts.context->rank;
  for (int i = 0; i < sccl_args.nranks; i++) {
    sccl_args.chip_map[i] = i;
  }
  tpudnnStatus_t ret = tpudnnC2CGather(
      opts.handle_, tpudnnPhysToVirt(opts.handle_, (uint64_t)opts.send_buff_), opts.input_elements,
      tpudnnPhysToVirt(opts.handle_, (uint64_t)opts.recv_buff_), opts.output_elements, opts.dtype_, opts.root, sccl_args);
  return;
}

}  // namespace sophon
