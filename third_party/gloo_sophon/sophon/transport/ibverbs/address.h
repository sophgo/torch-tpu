/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <infiniband/verbs.h>

#include "sophon/transport/address.h"

namespace sophon {
namespace transport {
namespace ibverbs {

// Forward declaration
class Pair;

class Address : public ::sophon::transport::Address {
 public:
  Address();
  explicit Address(const std::vector<char>&);
  virtual ~Address() {}

  virtual std::vector<char> bytes() const override;
  virtual std::string str() const override;

 protected:
  explicit Address(const Address&) = default;

  struct {
    uint32_t lid;
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid ibv_gid;
  } addr_;

  // Pair can access addr_ directly
  friend class Pair;
};

} // namespace ibverbs
} // namespace transport
} // namespace sophon
