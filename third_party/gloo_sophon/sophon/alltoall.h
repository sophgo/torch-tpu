/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "sophon/common/logging.h"
#include "sophon/context.h"
#include "sophon/transport/unbound_buffer.h"
#include "sophon_defines_2260.h"

namespace sophon {

class AlltoallOptions {
 public:
  explicit AlltoallOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()), chip_map_(context->chip_map) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->elements = buf->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->in = std::move(buf);
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
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

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    this->elements = elements;    
    this->elementSize = sizeof(T);
    this->out = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  void setOutputSophon(tpudnnHandle_t handle, void* send_buff, 
                       void* recv_buff, size_t bytes, tpudnnDataType_t sg_type) {
    this->handle_ = handle;
    this->send_buff_ = send_buff;
    this->recv_buff_ = recv_buff;
    this->bytes_ = bytes;
    this->sg_type_ = sg_type;
  } 

  void setTag(uint32_t tag) {
    this->tag = tag;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    SOPHON_ENFORCE(timeout.count() > 0);
    this->timeout = timeout;
  }

 protected:
  std::shared_ptr<Context> context;
  std::unique_ptr<transport::UnboundBuffer> in;
  std::unique_ptr<transport::UnboundBuffer> out;

  tpudnnHandle_t handle_;
  void* send_buff_;
  void* recv_buff_;
  size_t bytes_;
  tpudnnDataType_t sg_type_;

  // Number of elements.
  size_t elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  std::vector<int> chip_map_;

  friend void alltoall( AlltoallOptions&);
};

void alltoall(AlltoallOptions &opts);

} // namespace sophon
