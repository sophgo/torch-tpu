/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "sophon/context.h"
#include "sophon/transport/unbound_buffer.h"
#include "sophon_defines_2260.h"
#include "types.h"
#include "sccl.h"
#include <c10/util/Exception.h>

namespace sophon {

class AllgatherOptions {
 public:
  explicit AllgatherOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->input_elements = buf->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->in = std::move(buf);
  }

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    this->input_elements = elements;
    this->elementSize = sizeof(T);
    this->in = context->createUnboundBuffer(ptr, input_elements * sizeof(T));
  }

  template <typename T>
  void setOutput(std::unique_ptr<transport::UnboundBuffer> buf) {
    this->output_elements = buf->size / sizeof(T);
    this->elementSize = sizeof(T);
    this->out = std::move(buf);
  }

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    this->output_elements = elements;
    this->elementSize = sizeof(T);
    this->out = context->createUnboundBuffer(ptr, output_elements * sizeof(T));
  }

  void setTag(uint32_t tag) {
    this->tag = tag;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    this->timeout = timeout;
  }

 protected:
  std::shared_ptr<Context> context;
  std::unique_ptr<transport::UnboundBuffer> in;
  std::unique_ptr<transport::UnboundBuffer> out;

  // Number of elements.
  size_t input_elements = 0;
  size_t output_elements = 0;

  // Number of bytes per element.
  size_t elementSize = 0;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  friend void allgather(AllgatherOptions&);

  friend scclResult_t scclAllGather(const void *, void *, size_t,
                                      sg_data_type_t, scclComm_t,
                                      tpudnnHandle_t);
};

void allgather(AllgatherOptions &opts);

scclResult_t scclAllGather(const void *send_buff, void *recv_buff,
                             size_t send_count, sg_data_type_t dtype,
                             scclComm_t comm, tpudnnHandle_t handle);

} // namespace sophon
