/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sophon/common/error.h"
#include "sophon/common/store.h"
#include "sophon/context.h"
#include "sophon/rendezvous/store.h"
#include "sophon/transport/address.h"
#include "sophon/transport/device.h"

namespace sophon {
namespace rendezvous {

class ContextFactory;

class Context : public ::sophon::Context {
 public:
  Context(int rank, int size, int base = 2);
  virtual ~Context();

  void connectFullMesh(
      Store& store,
      std::shared_ptr<transport::Device>& dev);

 protected:
  friend class ContextFactory;
};

class ContextFactory {
 public:
  static constexpr auto kMaxAddressSize =
      ::sophon::transport::Address::kMaxByteSize;

  explicit ContextFactory(std::shared_ptr<::sophon::Context> backingContext);

  std::shared_ptr<::sophon::Context> makeContext(
    std::shared_ptr<transport::Device>& dev);

 protected:
  std::shared_ptr<::sophon::Context> backingContext_;

  std::vector<std::vector<char>> recvData_;
  std::vector<std::vector<char>> sendData_;

  std::vector<std::unique_ptr<transport::Buffer>> recvBuffers_;
  std::vector<std::unique_ptr<transport::Buffer>> sendBuffers_;

  std::vector<int> recvNotificationData_;
  std::vector<std::unique_ptr<transport::Buffer>> recvNotificationBuffers_;

  std::vector<int> sendNotificationData_;
  std::vector<std::unique_ptr<transport::Buffer>> sendNotificationBuffers_;
};


} // namespace rendezvous

} // namespace sophon
