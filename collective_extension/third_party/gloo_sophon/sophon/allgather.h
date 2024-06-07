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
#include <c10/util/Exception.h>

namespace sophon {

class AllgatherOptions {
 public:
  explicit AllgatherOptions(const std::shared_ptr<Context>& context)
      : context(context), timeout(context->getTimeout()), chip_map_(context->chip_map) {}

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

  template<typename T>
  void setInput(tpudnnHandle_t handle, void *buf, size_t elements) {
    this->input_elements = elements;
    this->elementSize = sizeof(T);
    this->send_buff_ = buf;
    this->handle_ = handle;
  }

  template<typename T>
  void setOutput(tpudnnHandle_t handle, void *buf, size_t elements) {
    this->output_elements = elements;
    this->elementSize = sizeof(T);
    this->recv_buff_ = buf;
    this->handle_ = handle;
    if (typeid(T) == typeid(float)) {
      this->dtype_ = SG_DTYPE_FP32;
    } else if (typeid(T) == typeid(sophon::float16)) {
      this->dtype_ = SG_DTYPE_FP16;
    } else if (typeid(T) == typeid(int8_t)) {
      this->dtype_ = SG_DTYPE_INT8;
    } else if (typeid(T) == typeid(uint8_t)) {
      this->dtype_ = SG_DTYPE_UINT8;
    } else if (typeid(T) == typeid(int32_t)) {
      this->dtype_ = SG_DTYPE_INT32;
    } else {
      TORCH_CHECK(false, "Invalid data type\n");
    }
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

  tpudnnHandle_t handle_;
  void* send_buff_;
  size_t send_bytes_;
  void* recv_buff_;
  size_t recv_bytes_;
  sg_data_type_t dtype_;
  std::vector<int> chip_map_;

  friend void allgather(AllgatherOptions&);

  friend void allgather2260(AllgatherOptions &);
};

void allgather(AllgatherOptions& opts);

void allgather2260(AllgatherOptions &opts);

} // namespace sophon
