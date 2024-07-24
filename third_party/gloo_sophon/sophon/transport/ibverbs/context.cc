/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/transport/ibverbs/context.h"

#include "sophon/common/error.h"
#include "sophon/transport/ibverbs/device.h"
#include "sophon/transport/ibverbs/pair.h"

namespace sophon {
namespace transport {
namespace ibverbs {

Context::Context(std::shared_ptr<Device> device, int rank, int size)
    : ::sophon::transport::Context(rank, size), device_(device) {}

Context::~Context() {}

std::unique_ptr<transport::Pair>& Context::createPair(int rank) {
  pairs_[rank] = std::unique_ptr<transport::Pair>(
      new ibverbs::Pair(device_, getTimeout()));
  return pairs_[rank];
}

std::unique_ptr<transport::UnboundBuffer> Context::createUnboundBuffer(
    void* ptr,
    size_t size) {
  SOPHON_THROW_INVALID_OPERATION_EXCEPTION(
      "Unbound buffers not supported yet for ibverbs transport");
  return std::unique_ptr<transport::UnboundBuffer>();
}

} // namespace ibverbs
} // namespace transport
} // namespace sophon
