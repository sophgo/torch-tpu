#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "torch_tpu/csrc/core/TPUGuardImpl.h"
#include "torch_tpu/csrc/core/TPUStream.h"
#include <cstddef>

using namespace c10;

namespace c10_tpu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for TPU.  It accepts
/// integer indices (interpreting them as TPU devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// cudaSetDevice/cudaGetDevice calls); however, it can only be used
/// from code that links against TPU directly.
struct TPUGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit TPUGuard() = delete;

  /// Set the current TPU device to the passed device index.
  explicit TPUGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current TPU device to the passed device.  Errors if the passed
  /// device is not a TPU device.
  explicit TPUGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  TPUGuard(const TPUGuard&) = delete;
  TPUGuard& operator=(const TPUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  TPUGuard(TPUGuard&& other) = delete;
  TPUGuard& operator=(TPUGuard&& other) = delete;

  /// Sets the TPU device to the given device.  Errors if the given device
  /// is not a TPU device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the TPU device to the given device.  Errors if the given device
  /// is not a TPU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the TPU device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<c10_tpu::impl::TPUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for TPU.  See
/// TPUGuard for when you can use this.
struct OptionalTPUGuard {
  /// Create an uninitialized OptionalTPUGuard.
  explicit OptionalTPUGuard() : guard_() {}

  /// Set the current TPU device to the passed Device, if it is not nullopt.
  explicit OptionalTPUGuard(optional<Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current TPU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalTPUGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalTPUGuard(const OptionalTPUGuard&) = delete;
  OptionalTPUGuard& operator=(const OptionalTPUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalTPUGuard(OptionalTPUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalTPUGuard& operator=(OptionalTPUGuard&& other) = delete;

  /// Sets the TPU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a TPU
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the TPU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a TPU device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the TPU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original TPU device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<c10_tpu::impl::TPUGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for TPU.  See TPUGuard
/// for when you can use this.
struct TPUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit TPUStreamGuard() = delete;

  /// Set the current TPU device to the device associated with the passed
  /// stream, and set the current TPU stream on that device to the passed
  /// stream. Errors if the Stream is not a TPU stream.
  explicit TPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  TPUStreamGuard(const TPUStreamGuard&) = delete;
  TPUStreamGuard& operator=(const TPUStreamGuard&) = delete;

  /// Move is disallowed, as TPUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  TPUStreamGuard(TPUStreamGuard&& other) = delete;
  TPUStreamGuard& operator=(TPUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a TPU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on TPU, use TPUMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the TPU stream that was set at the time the guard was
  /// constructed.
  TPUStream original_stream() const {
    return TPUStream(TPUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent TPU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  TPUStream current_stream() const {
    return TPUStream(TPUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent TPU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the TPU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<c10_tpu::impl::TPUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for TPU.  See
/// TPUGuard for when you can use this.
struct OptionalTPUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalTPUStreamGuard() : guard_() {}

  /// Set the current TPU device to the device associated with the passed
  /// stream, and set the current TPU stream on that device to the passed
  /// stream. Errors if the Stream is not a TPU stream.
  explicit OptionalTPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalTPUStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalTPUStreamGuard(const OptionalTPUStreamGuard&) = delete;
  OptionalTPUStreamGuard& operator=(const OptionalTPUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalTPUStreamGuard(OptionalTPUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalTPUStreamGuard& operator=(OptionalTPUStreamGuard&& other) = delete;

  /// Resets the currently set TPU stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the TPU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  optional<TPUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(TPUStream(TPUStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent TPU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<TPUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(TPUStream(TPUStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original TPU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<c10_tpu::impl::TPUGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for TPU.
struct TPUMultiStreamGuard {
  explicit TPUMultiStreamGuard(ArrayRef<TPUStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  TPUMultiStreamGuard(const TPUMultiStreamGuard&) = delete;
  TPUMultiStreamGuard& operator=(const TPUMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  TPUMultiStreamGuard(TPUMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  TPUMultiStreamGuard& operator=(TPUMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<c10_tpu::impl::TPUGuardImpl> guard_;

  static std::vector<Stream> unwrapStreams(ArrayRef<TPUStream> tpuStreams) {
    std::vector<Stream> streams;
    streams.reserve(tpuStreams.size());
    for (const TPUStream& tpuStream : tpuStreams) {
      streams.push_back(tpuStream);
    }
    return streams;
  }
};


} // namespace c10_tpu
