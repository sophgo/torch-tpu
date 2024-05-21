/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "sophon/context.h"
#include "sophon/transport/unbound_buffer.h"
#include "sophon_defines_2260.h"

namespace sophon {

class ScatterOptions {
 public:
  explicit ScatterOptions(const std::shared_ptr<Context>& context)
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

  void setOutputSophon(tpudnnHandle_t handle, void* send_buff,
                       size_t send_bytes, void* recv_buff,
                       size_t recv_bytes, sg_data_type_t sg_type) {
    this->handle_ = handle;
    this->send_buff_ = send_buff;
    this->send_bytes_ = send_bytes;
    this->recv_buff_ = recv_buff;
    this->recv_bytes_ = recv_bytes;
    this->dtype_ = sg_type;
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

  void setRoot(int root) { this->root = root; }

  void setTag(uint32_t tag) { this->tag = tag; }

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

  // Rank of process to scatter from.
  int root = -1;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  tpudnnHandle_t handle_;
  void* send_buff_;
  size_t send_bytes_;
  void* recv_buff_;
  size_t recv_bytes_;
  sg_data_type_t dtype_;

  friend void scatter(ScatterOptions&);

  friend void scatter2260(ScatterOptions &);
};

void scatter(ScatterOptions& opts);

void scatter2260(ScatterOptions& opts);

}  // namespace sophon
