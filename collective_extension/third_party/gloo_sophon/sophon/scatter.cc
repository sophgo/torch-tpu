/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/scatter.h"

#include <algorithm>
#include <cstring>

#include "sophon/common/logging.h"
#include "sophon/types.h"

namespace sophon {

void scatter(ScatterOptions& opts) {
    // TO DO
}

void scatter2260(ScatterOptions &opts) {
  // call tpudnnC2CScatter
  sccl_args_t sccl_args = {0};
  sccl_args.nranks = opts.context->size;
  sccl_args.rank = opts.context->rank;
  if (opts.chip_map_.empty()) {
    for (int i = 0; i < sccl_args.nranks; i++) {
      sccl_args.chip_map[i] = i;
    }
  } else {
    memcpy(sccl_args.chip_map, opts.chip_map_.data(),
           sizeof(opts.chip_map_.size()) * 4);
  }

  tpudnnStatus_t ret = tpudnnC2CScatter(
      opts.handle_, tpudnnPhysToVirt(opts.handle_, (uint64_t)opts.send_buff_), opts.input_elements, opts.dtype_,
      tpudnnPhysToVirt(opts.handle_, (uint64_t)opts.recv_buff_), opts.output_elements, opts.dtype_, opts.root, sccl_args);
  return;
}


}  // namespace sophon
