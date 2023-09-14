/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <exception>

#include "sophon/common/string.h"

#define SOPHON_ERROR_MSG(...) \
  ::sophon::MakeString("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__)

namespace sophon {

const std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds::zero();

// A base class for all sophon runtime errors
struct Exception : public std::runtime_error {
  Exception() = delete;
  explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

#define SOPHON_THROW(...) \
  throw ::sophon::Exception(SOPHON_ERROR_MSG(__VA_ARGS__))


// Thrown for invalid operations on sophon APIs
struct InvalidOperationException : public ::sophon::Exception {
  InvalidOperationException() = delete;
  explicit InvalidOperationException(const std::string& msg)
      : ::sophon::Exception(msg) {}
};

#define SOPHON_THROW_INVALID_OPERATION_EXCEPTION(...) \
  throw ::sophon::InvalidOperationException(SOPHON_ERROR_MSG(__VA_ARGS__))


// Thrown for unrecoverable IO errors
struct IoException : public ::sophon::Exception {
  IoException() = delete;
  explicit IoException(const std::string& msg) : ::sophon::Exception(msg) {}
};

#define SOPHON_THROW_IO_EXCEPTION(...) \
  throw ::sophon::IoException(SOPHON_ERROR_MSG(__VA_ARGS__))

} // namespace sophon
