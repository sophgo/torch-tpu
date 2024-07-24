/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/context.h"

#include "sophon/common/error.h"
#include "sophon/common/logging.h"
#include "sophon/transport/device.h"
#include "sophon/transport/unbound_buffer.h"

namespace sophon {

static const std::chrono::seconds kTimeoutDefault = std::chrono::seconds(30);

Context::Context(int rank, int size, int base)
    : rank(rank),
      size(size),
      base(base),
      slot_(0),
      timeout_(kTimeoutDefault) {
  SOPHON_ENFORCE_GE(rank, 0);
  SOPHON_ENFORCE_LT(rank, size);
  SOPHON_ENFORCE_GE(size, 1);
}

Context::~Context() {
}

std::shared_ptr<transport::Device>& Context::getDevice() {
  SOPHON_ENFORCE(device_, "Device not set!");
  return device_;
}

std::unique_ptr<transport::Pair>& Context::getPair(int i) {
  SOPHON_ENFORCE(transportContext_, "Transport context not set!");
  return transportContext_->getPair(i);
}

std::unique_ptr<transport::UnboundBuffer> Context::createUnboundBuffer(
    void* ptr, size_t size) {
  return transportContext_->createUnboundBuffer(ptr, size);
}

int Context::nextSlot(int numToSkip) {
  SOPHON_ENFORCE_GT(numToSkip, 0);
  auto temp = slot_;
  slot_ += numToSkip;
  return temp;
}

void Context::closeConnections() {
  for (auto i = 0; i < size; i++) {
    auto& pair = getPair(i);
    if (pair) {
      pair->close();
    }
  }
}

void Context::setTimeout(std::chrono::milliseconds timeout=kTimeoutDefault) {
  SOPHON_ENFORCE(timeout.count() >= 0, "Invalid timeout");
  timeout_ = timeout;
}

std::chrono::milliseconds Context::getTimeout() const {
  return timeout_;
}

} // namespace sophon
