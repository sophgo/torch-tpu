#pragma once

#include <cstdint>
#include <mutex>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/SmallVector.h>

#ifdef BACKEND_SG2260
#include "torch_tpu/csrc/core/Interface/sgrtInterface.h"
#endif

#include <tpuDNN.h>
/*
 * Stream pool note.
 *
 * A TPUStream is an abstraction of an actual sgStream on the TPU. TPUStreams
 * are backed by sgStreams, but they use several pools to minimize the costs
 * associated with creating, retaining, and destroying sgStreams.
 *
 * There are three pools per device, and a device's pools are lazily created.
 *
 * The first pool contains only the default stream. When the default stream
 * is requested it's returned.
 *
 * The second pool is the "low priority" or "default priority" streams. In
 * HIP builds there is no distinction between streams in this pool and streams
 * in the third pool (below). There are 32 of these streams per device, and
 * when a stream is requested one of these streams is returned round-robin.
 * That is, the first stream requested is at index 0, the second at index 1...
 * to index 31, then index 0 again.
 *
 * This means that if 33 low priority streams are requested, the first and
 * last streams requested are actually the same stream (under the covers)
 * and kernels enqueued on them cannot run concurrently.
 *
 * The third pool is the "high priority" streams. The third pool acts like
 * the second pool except the streams are created with a higher priority.
 *
 * These pools suggest that stream users should prefer many short-lived streams,
 * as the cost of acquiring and releasing streams is effectively zero. If
 * many longer-lived streams are required in performance critical scenarios
 * then the functionality here may need to be extended to allow, for example,
 * "reserving" a subset of the pool so that other streams do not accidentally
 * overlap the performance critical streams.
 *
 * Note: although the notion of "current stream for device" is thread local
 * (every OS thread has a separate current stream, as one might expect),
 * the stream pool is global across all threads; stream 0 is always stream 0
 * no matter which thread you use it on.  Multiple threads can synchronize
 * on the same stream.  Although the TPU documentation is not very clear
 * on the matter, streams are thread safe; e.g., it is safe to enqueue
 * a kernel on the same stream from two different threads.
 */
namespace c10_tpu {

static constexpr int max_compile_time_stream_priorities = 4;

// Value object representing a TPU stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra TPU-specific
// functionality (conversion to sgrtStream_t), and a guarantee that
// the wrapped c10::Stream really is a TPU stream.
class TPUStream {
public:
  enum Unchecked { UNCHECKED };

  explicit TPUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1);
  }

  /// Construct a TPUStream from a Stream with no error checking.
  /// This constructor uses the "named" constructor idiom, and can
  /// be invoked as: TPUStream(TPUStream::UNCHECKED, stream)
  explicit TPUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

  bool operator==(const TPUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const TPUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

#ifdef BACKEND_SG2260
  /// Implicit conversion to sgrtStream_t.
  operator sgrt::sgrtStream_t() const {
    return stream();
  }

  bool query() const {
    c10::DeviceGuard guard{stream_.device()};
    sgrt::sgrtStreamStatus status = sgrt::SgrtStreamQuery(stream());
    if ( status == sgrt::SG_STREAM_STATUS_COMPLETE ) {
      return true;
    }
    return false;
  }

  void synchronize() const {
    c10::DeviceGuard guard{stream_.device()};
    sgrt::SgrtSynchronizeStream(stream());
  }

  /// Explicit conversion to rtStream_t.
  sgrt::sgrtStream_t stream() const;

#endif

  operator tpudnnHandle_t() const;

  /// Implicit conversion to pytorch Stream.
  operator c10::Stream() const {
    return unwrap();
  }

  /// Used to avoid baking in device type explicitly to Python-side API.
  c10::DeviceType device_type() const {
    return c10::DeviceType::PrivateUse1;
  }

  /// Get the TPU device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a TPU device.
  c10::Device device() const {
    return c10::Device(c10::DeviceType::PrivateUse1, device_index());
  }

  c10::StreamId id() const {
    return stream_.id();
  }

  /// Explicit conversion to Stream.
  c10::Stream unwrap() const {
    return stream_;
  }

   /// The TPUStream can be unpacked using unpack().
  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  // Unpack a TPUStream from the 3 fields generated by pack().
  static TPUStream unpack3(
      c10::StreamId stream_id,
      c10::DeviceIndex device_index,
      c10::DeviceType device_type) {
    return TPUStream(c10::Stream::unpack3(stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range() {
    // Note: this returns the range of priority **supported by PyTorch**
    int least_priority = 0, greatest_priority = c10_tpu::max_compile_time_stream_priorities;
    return std::make_tuple(least_priority, greatest_priority);
  }

private:
  c10::Stream stream_;
};

/**
 * Get a new stream from the TPU stream pool.  You can think of this
 * as "creating" a new stream, but no such creation actually happens;
 * instead, streams are preallocated from the pool and returned in a
 * round-robin fashion.
 *
 * You can request a stream from the high priority pool by setting
 * isHighPriority to true, or a stream for a specific device by setting device
 * (defaulting to the current TPU stream.)
 */
TPUStream getStreamFromPool(const bool isHighPriority = false, c10::DeviceIndex device = -1);

// no default priority to disambiguate overloads
TPUStream getStreamFromPool(const int priority, c10::DeviceIndex device = -1);

/**
 * Get a TPUStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
#ifdef BACKEND_SG2260
TPUStream getStreamFromExternal(sgrt::sgrtStream_t ext_stream, c10::DeviceIndex device_index);
#endif


/**
 * Get the default TPU stream, for the passed TPU device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
TPUStream getDefaultTPUStream(c10::DeviceIndex device_index = -1);

/**
 * Get the current TPU stream, for the passed TPU device, or for the
 * current device if no device index is passed.  The current TPU stream
 * will usually be the default TPU stream for the device, but it may
 * be different if someone called 'setCurrentTPUStream' or used 'StreamGuard'
 * or 'TPUStreamGuard'.
 */
TPUStream getCurrentTPUStream(c10::DeviceIndex device_index = -1);

/**
 * Set the current stream on the device of the passed in stream to be
 * the passed in stream.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * stream of the device of the passed stream.
 *
 * Confused?  Avoid using this function; prefer using 'TPUStreamGuard' instead
 * (which will switch both your current device and current stream in the way you
 * expect, and reset it back to its original state afterwards).
 */
void setCurrentTPUStream(TPUStream stream);

/**
 * wait all kernels on all devices finish.
*/
void tpuSynchronizeDevice();

std::ostream& operator<<(std::ostream& stream, const TPUStream& s);

} // namespace c10_tpu

namespace std {
template <>
struct hash<c10_tpu::TPUStream> {
  size_t operator()(c10_tpu::TPUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
