/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>

#include "sophon/context.h"
#include "sophon/transport/unbound_buffer.h"
#include "sophon_defines_2260.h"

namespace sophon {

class ReduceOptions {
public:
  using Func = std::function<void(void *, const void *, const void *, size_t)>;

  explicit ReduceOptions(const std::shared_ptr<Context> &context)
      : context(context), timeout(context->getTimeout()), chip_map_(context->chip_map) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->elements = buf->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->in = std::move(buf);
  }

  template <typename T>
  void setInput(T *ptr, size_t elements) {
    this->elements = elements;
    this->elementSize = sizeof(T);
    this->in = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->elements = buf->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->out = std::move(buf);
  }

  void setOutputSophon(tpudnnHandle_t handle, void* send_buff,
                       void* recv_buff, size_t bytes,
                       sg_data_type_t sg_type, sg_reduce_method_t reduce_method) {
    this->handle_ = handle;
    this->send_buff_ = send_buff;
    this->recv_buff_ = recv_buff;
    this->bytes_ = bytes;
    this->dtype_ = sg_type;
    this->reduce_method_ = reduce_method;
  }

  template <typename T>
  void setOutput(T *ptr, size_t elements) {
    this->elements = elements;
    this->elementSize = sizeof(T);
    this->out = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  void setRoot(int root) { this->root = root; }

  void setReduceFunction(Func fn) { this->reduce = fn; }

  void setTag(uint32_t tag) { this->tag = tag; }

  void setMaxSegmentSize(size_t maxSegmentSize) {
    this->maxSegmentSize = maxSegmentSize;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    this->timeout = timeout;
  }

protected:
  std::shared_ptr<Context> context;

  tpudnnHandle_t handle_;
  void* send_buff_;
  void* recv_buff_;
  size_t bytes_;
  sg_data_type_t dtype_;
  sg_reduce_method_t reduce_method_;

  std::unique_ptr<transport::UnboundBuffer> in;
  std::unique_ptr<transport::UnboundBuffer> out;

  // Number of elements.
  size_t elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Rank of process to reduce to.
  int root = -1;

  // Reduction function.
  Func reduce;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // This is the maximum size of each I/O operation (send/recv) of which
  // two are in flight at all times. A smaller value leads to more
  // overhead and a larger value leads to poor cache behavior.
  static constexpr size_t kMaxSegmentSize = 1024 * 1024;

  // Internal use only. This is used to exercise code paths where we
  // have more than 2 segments per rank without making the tests slow
  // (because they would require millions of elements if the default
  // were not configurable).
  size_t maxSegmentSize = kMaxSegmentSize;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  std::vector<int> chip_map_;

  friend void reduce(ReduceOptions &);
};

void reduce(ReduceOptions &opts);

} // namespace sophon
