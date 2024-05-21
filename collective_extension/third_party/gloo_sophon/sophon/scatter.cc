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
  sccl_args_t sccl_args;
  tpudnnStatus_t ret = tpudnnC2CScatter(
      opts.handle_, opts.send_buff_, opts.input_elements, opts.dtype_,
      opts.recv_buff_, opts.output_elements, opts.dtype_, opts.root, sccl_args);
  return;
}


}  // namespace sophon
