/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "sophon/common/memory.h"
#include "sophon/common/store.h"
#include "sophon/transport/context.h"

namespace sophon {
namespace transport {
namespace tcp {

// Forward declaration
class Context;
class Device;
class Pair;
class UnboundBuffer;

class Context : public ::sophon::transport::Context,
                      public std::enable_shared_from_this<Context> {
 public:
  Context(std::shared_ptr<Device> device, int rank, int size);

  virtual ~Context();

  virtual void createAndConnectAllPairs(IStore& store) override;

  std::unique_ptr<transport::Pair>& createPair(int rank) override;

  std::unique_ptr<transport::UnboundBuffer> createUnboundBuffer(
      void* ptr,
      size_t size) override;

 protected:
  std::shared_ptr<Device> device_;

  using pendingRecvTuple = std::tuple<
      WeakNonOwningPtr<UnboundBuffer>,
      size_t,
      size_t,
      std::unordered_set<int>>;

  // Buffers with pending receive operation by slot.
  std::unordered_map<uint64_t, std::deque<pendingRecvTuple>> pendingRecv_;

  // This function registers the specified unbound buffer for a receive
  // operation from any of the specified ranks.
  void recvFromAny(
      UnboundBuffer* buf,
      uint64_t slot,
      size_t offset,
      size_t nbytes,
      std::vector<int> srcRanks);

  int recvFromAnyFindRank(
      UnboundBuffer* buf,
      uint64_t slot,
      size_t offset,
      size_t nbytes,
      const std::vector<int>& srcRanks);

  // Allowed to be called only by ContextMutator::findRecvFromAny,
  // where the context lock is already held.
  bool findRecvFromAny(
      uint64_t slot,
      int rank,
      WeakNonOwningPtr<tcp::UnboundBuffer>* buf,
      size_t* offset,
      size_t* nbytes);

  // Set exception on every pair in this context. This is called when
  // waiting for a send or recv operation on an unbound buffer times
  // out. All pairs should be signaled and closed in that event.
  void signalException(const std::string& msg);

  friend class ContextMutator;

  friend class UnboundBuffer;

  friend class Pair;
};

struct Rank {
  std::string hostname;
  std::vector<char> addressBytes;
  std::vector<ssize_t> pairIdentifiers;

  explicit Rank(
      const std::string& hostname,
      const std::vector<char>& addrBytes,
      const std::vector<ssize_t>& pairIdentifiers)
      : hostname(hostname),
        addressBytes(addrBytes),
        pairIdentifiers(pairIdentifiers) {}
  explicit Rank(const std::vector<char>& bytes);

  std::vector<char> bytes() const;
};

} // namespace tcp
} // namespace transport
} // namespace sophon
