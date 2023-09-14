/**
 * Copyright (c) 2020-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sophon/transport/tcp/tls/pair.h"

#include "sophon/common/error.h"
#include "sophon/common/logging.h"
#include "sophon/transport/tcp/buffer.h"
#include "sophon/transport/tcp/tls/context.h"
#include "sophon/transport/tcp/tls/device.h"
#include "sophon/transport/tcp/unbound_buffer.h"

#include <cstring>
#include <poll.h>

namespace sophon {
namespace transport {
namespace tcp {
namespace tls {

Pair::Pair(Context *context, Device *device, int rank,
           std::chrono::milliseconds timeout)
    : ::sophon::transport::tcp::Pair(context, device, rank, timeout),
      ssl_(nullptr),
      ssl_ctx_(dynamic_cast<Context *>(context_)->ssl_ctx_.get()),
      is_ssl_connected_(false), fatal_error_occurred_(false) {}

Pair::~Pair() {
  std::lock_guard<std::mutex> lock(m_);
  if (state_ != CLOSED) {
    Pair::changeState(CLOSED);
  }
}

int Pair::handshake() {
  SOPHON_ENFORCE(state_ == CONNECTED);
  SOPHON_ENFORCE(!is_ssl_connected_);
  int r = _sophontls::SSL_do_handshake(ssl_);
  if (r == 1) {
    is_ssl_connected_ = true;
    cv_.notify_all();
    return 0;
  }
  int err = _sophontls::SSL_get_error(ssl_, r);
  int events = 0;
  if (err == SSL_ERROR_WANT_WRITE) {
    events = POLLOUT | POLLERR;
  } else if (err == SSL_ERROR_WANT_READ) {
    events = POLLIN | POLLERR;
  } else {
    signalException(
        SOPHON_ERROR_MSG("SSL_do_handshake failed: " + getSSLErrorMessage()));
  }
  return events;
}

bool Pair::write(Op &op) {
  NonOwningPtr<UnboundBuffer> buf;
  std::array<struct iovec, 2> iov;
  int ioc;

  const auto opcode = op.getOpcode();

  // Acquire pointer to unbound buffer if applicable.
  if (opcode == Op::SEND_UNBOUND_BUFFER) {
    buf = NonOwningPtr<UnboundBuffer>(op.ubuf);
    if (!buf) {
      return false;
    }
  }

  const auto nbytes = prepareWrite(op, buf, iov.data(), ioc);
  ssize_t total_rv = 0;
  for (int i = 0; i < ioc; ++i) {
    for (;;) {
      if (iov[i].iov_len == 0) {
        break;
      }
      ssize_t rv = _sophontls::SSL_write(ssl_, iov[i].iov_base, iov[i].iov_len);
      if (rv <= 0) {
        int err = _sophontls::SSL_get_error(ssl_, rv);

        SOPHON_ENFORCE(err != SSL_ERROR_NONE);
        SOPHON_ENFORCE(err != SSL_ERROR_WANT_READ);

        if (err == SSL_ERROR_WANT_WRITE) {
          // just repeat the same write
          continue;
        }

        if (err == SSL_ERROR_SYSCALL) {
          fatal_error_occurred_ = true;
          if (errno == EPIPE) {
            if (!sync_) {
              return false;
            }
          }
        }

        // Unexpected error
        signalException(SOPHON_ERROR_MSG(
            "SSL_write ", peer_.str(), " failed: ", "ssl error: ", err,
            ", errno = ", strerror(errno),
            ", ssl error message: ", getSSLErrorMessage()));
        return false;
      }
      total_rv += rv;
      op.nwritten += rv;
      break;
    }
  }
  SOPHON_ENFORCE_EQ(total_rv, nbytes);
  SOPHON_ENFORCE_EQ(op.nwritten, op.preamble.nbytes);

  writeComplete(op, buf, opcode);
  return true;
}

bool Pair::read() {
  NonOwningPtr<UnboundBuffer> buf;

  for (;;) {
    struct iovec iov = {
        .iov_base = nullptr,
        .iov_len = 0,
    };
    const auto nbytes = prepareRead(rx_, buf, iov);
    if (nbytes < 0) {
      return false;
    }

    // Break from loop if the op is complete.
    // Note that this means that the buffer pointer has been
    // set, per the call to prepareRead.
    if (nbytes == 0) {
      break;
    }

    ssize_t rv = 0;
    for (;;) {
      rv = _sophontls::SSL_read(ssl_, iov.iov_base, iov.iov_len);
      if (rv <= 0) {
        int err = _sophontls::SSL_get_error(ssl_, rv);

        SOPHON_ENFORCE(err != SSL_ERROR_NONE);
        SOPHON_ENFORCE(err != SSL_ERROR_WANT_WRITE);

        if (err == SSL_ERROR_WANT_READ) {
          return false;
        }

        if (err == SSL_ERROR_ZERO_RETURN) {
          if (!sync_) {
            return false;
          }
        }

        if (err == SSL_ERROR_SYSCALL) {
          fatal_error_occurred_ = true;
          if (errno == EPIPE || errno == ECONNRESET) {
            if (!sync_) {
              return false;
            }
          }
        }

        // Unexpected error
        signalException(SOPHON_ERROR_MSG(
            "SSL_read ", peer_.str(), " failed: ", "ssl error: ", err,
            ", errno = ", strerror(errno),
            ", ssl error message: ", getSSLErrorMessage()));
        return false;
      }
      break;
    }

    // Transition to CLOSED on EOF
    if (rv == 0) {
      signalException(
          SOPHON_ERROR_MSG("Connection closed by peer ", peer_.str()));
      return false;
    }

    rx_.nread += rv;
  }

  readComplete(buf);
  return true;
}

void Pair::handleReadWrite(int events) {
  if (!is_ssl_connected_ && !device_->isInitiator(self_, peer_)) {
    if (ssl_ == nullptr) {
      SOPHON_ENFORCE(ssl_ctx_ != nullptr);
      ssl_ = _sophontls::SSL_new(ssl_ctx_);
      SOPHON_ENFORCE(ssl_ != nullptr, getSSLErrorMessage());
      SOPHON_ENFORCE(_sophontls::SSL_set_fd(ssl_, fd_) == 1, getSSLErrorMessage());
      _sophontls::SSL_set_accept_state(ssl_);
    }
    int es;
    if ((es = handshake())) {
      device_->registerDescriptor(fd_, es, this);
    }
  } else {
    tcp::Pair::handleReadWrite(events);
  }
}

void Pair::changeState(Pair::state nextState) noexcept {
  if (nextState == CLOSED && is_ssl_connected_) {
    if (!fatal_error_occurred_) {
      if (_sophontls::SSL_shutdown(ssl_) == 0) {
        _sophontls::SSL_shutdown(ssl_);
      }
    }
    _sophontls::SSL_free(ssl_);
    ssl_ = nullptr;
    is_ssl_connected_ = false;
  }
  ::sophon::transport::tcp::Pair::changeState(nextState);
}

void Pair::waitUntilSSLConnected(std::unique_lock<std::mutex> &lock,
                                 bool useTimeout) {
  auto pred = [&] {
    throwIfException();
    return is_ssl_connected_;
  };
  waitUntil(pred, lock, useTimeout);
}

void Pair::waitUntilConnected(std::unique_lock<std::mutex> &lock,
                              bool useTimeout) {
  ::sophon::transport::tcp::Pair::waitUntilConnected(lock, useTimeout);

  if (!is_ssl_connected_) {
    if (device_->isInitiator(self_, peer_)) {
      SOPHON_ENFORCE(ssl_ == nullptr);
      SOPHON_ENFORCE(ssl_ctx_ != nullptr);
      ssl_ = _sophontls::SSL_new(ssl_ctx_);
      SOPHON_ENFORCE(ssl_ != nullptr, getSSLErrorMessage());
      SOPHON_ENFORCE(_sophontls::SSL_set_fd(ssl_, fd_) == 1, getSSLErrorMessage());
      _sophontls::SSL_set_connect_state(ssl_);
      int events;
      const int maxAttempts = 100;
      for (int j = 0; j < maxAttempts && (events = handshake()); j++) {
        int r = 0;
        struct pollfd pfd;
        pfd.fd = fd_;
        pfd.events = events;
        // Wait at most 100*100 ms for socket event
        for (int i = 0; i < 100 && r == 0; i++) {
          r = poll(&pfd, 1, 100);
        }
        // poll returns -1 on error
        SOPHON_ENFORCE(r == 1, "poll return ", r, ", error events: ", pfd.revents,
                     ", errno ", errno, " ", strerror(errno));
      }
      SOPHON_ENFORCE(is_ssl_connected_, "handshake was not succeeded after ",
                   maxAttempts, " attempts");
    } else {
      waitUntilSSLConnected(lock, useTimeout);
    }
  }
}

void Pair::verifyConnected() {
  ::sophon::transport::tcp::Pair::verifyConnected();
  SOPHON_ENFORCE(is_ssl_connected_, "Pair is not SSL connected (", self_.str(),
               " <--> ", peer_.str(), ")");
}

} // namespace tls
} // namespace tcp
} // namespace transport
} // namespace sophon
