/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/transport/ibverbs/memory_region.h"

#include <string.h>

#include "sophon/common/logging.h"

namespace sophon {
namespace transport {
namespace ibverbs {

MemoryRegion::MemoryRegion(struct ibv_pd* pd) {
  memset(&src_, 0, sizeof(src_));

  // Map this region so it can be used as source for a send, or as a
  // target for a receive.
  mr_ = ibv_reg_mr(pd, &src_, sizeof(src_), IBV_ACCESS_LOCAL_WRITE);
  SOPHON_ENFORCE(mr_);
}

MemoryRegion::MemoryRegion(struct ibv_pd* pd, struct ibv_mr* src)
    : MemoryRegion(pd) {
  memcpy(&src_, src, sizeof(src_));
}

MemoryRegion::~MemoryRegion() {
  int rv;

  rv = ibv_dereg_mr(mr_);
  SOPHON_ENFORCE_EQ(rv, 0);
}

} // namespace ibverbs
} // namespace transport
} // namespace sophon
