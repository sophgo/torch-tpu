#include <array>
#include <climits>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

#include <c10/core/Stream.h>
#include <c10/util/CallOnce.h>

#include "torch_tpu/csrc/core/TPUStream.h"
#include "torch_tpu/csrc/core/TPUFunction.h"
#include "torch_tpu/csrc/core/TPUGuard.h"
#include "torch_tpu/csrc/core/TPUException.h"
#include "TPUMacros.h"

#define tpuStreamNonBlocking 0x01 /**< Stream does not synchronize with stream 0 (the NULL stream) */

extern tpu_module_t get_kernel_module(tpu_resource_t stream);

namespace c10_tpu {
namespace {

// Global stream state and constants
static c10::once_flag init_flag;
static c10::DeviceIndex num_tpus = -1;
static constexpr int kStreamsPerPoolBits = 3;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr unsigned int kDefaultFlags = tpuStreamNonBlocking;
static constexpr int kStreamTypeBits = 4;

static std::once_flag device_flags[C10_COMPILE_TIME_MAX_TPUS];
static std::atomic<uint32_t> 
        priority_counters[c10_tpu::max_compile_time_stream_priorities][C10_COMPILE_TIME_MAX_TPUS];

#ifdef BACKEND_SG2260
static sgrt::sgrtStream_t 
        streams[c10_tpu::max_compile_time_stream_priorities][C10_COMPILE_TIME_MAX_TPUS][kStreamsPerPool];
static sgrt::sgrtStream_t
        default_streams[C10_COMPILE_TIME_MAX_TPUS] = {nullptr};
#endif

static tpudnnHandle_t tpudnn_handles[C10_COMPILE_TIME_MAX_TPUS] = {0};

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a sgrtStream_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams

class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const {
    return EXT == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

static inline StreamIdType streamIdType(c10::StreamId s) {
  // Externally allocated streams have their id being the cudaStream_ptr
  // so the last bit will be 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // last bit is external/internal stream, the mask should start from second
  // rightmost bit
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

static inline size_t streamIdIndex(c10::StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

// Thread-local current streams
static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

static void initGlobalStreamState() {
  num_tpus = device_count();
  // Check if the number of TPUs matches the expected compile-time max number
  // of TPUs.
  AT_ASSERTM(
      num_tpus <= C10_COMPILE_TIME_MAX_TPUS,
      "Number of TPU devices on the machine is larger than the compiled "
      "max number of tpus expected (",
      C10_COMPILE_TIME_MAX_TPUS,
      "). Increase that and recompile.");
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(c10::DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  TPUGuard device_guard{device_index};

#if 0
  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    for (int priority_i = 0; priority_i < c10_tpu::max_compile_time_stream_priorities; priority_i++)
    {
      auto& tpu_streami = streams[priority_i][device_index][i];
      auto pri = -priority_i; // lower number is higher priority

      C10_TPU_CHECK(sgrt::SgrtCreateStreamWithPriority(&tpu_streami, kDefaultFlags, pri));
      //TODO: trace to impl
    }
  }
#endif

#ifdef BACKEND_SG2260
  C10_TPU_CHECK(sgrt::SgrtCreateStream(&default_streams[device_index]));
  std::cout << "Stream created for device " << int(device_index) << std::endl;
#endif

  // The goal is to remove all runtime dependencies in torch-tpu,
  // then we can use tpudnnCreate to create tpuDNN handle. Util then
  // we have to keep all the resource management logic around.
  //
  // And don't bother freeing tpuDNN handles.
#ifdef BACKEND_SG2260
  auto stream = default_streams[device_index];
#else
  auto stream = tpu::TPUGetDeviceResource();
#endif
  auto kmodule = get_kernel_module(stream);
  tpudnn_handles[device_index] = tpudnnHandleFromStream(device_index, stream, kmodule);
}

static void initTPUStreamsOnce() {
  // Inits default and secondary streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  // Because tpurt does not have a default stream, we have to
  // add device init once any stream is used.
  //
  // We probably should not do this
  // Later when we implement priority streams we should reconsider
  DeviceIndex dev_index = current_device();
  std::call_once(
    device_flags[dev_index], initDeviceStreamState, dev_index);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<StreamId[]>(num_tpus);
  for (const auto i : c10::irange(num_tpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

static inline void check_tpu(c10::DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_tpus);
}

static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

TPUStream TPUStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return TPUStream(
      TPUStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::PrivateUse1, device_index),
          stream_id));
}

} // namespace

  TPUStream::operator tpudnnHandle_t() const
  {
    c10::DeviceIndex device_index = stream_.device_index();
    StreamId stream_id = stream_.id();
    StreamIdType st = streamIdType(stream_id);

    TORCH_INTERNAL_ASSERT(st.isDefault());

    return tpudnn_handles[device_index];
  }

#ifdef BACKEND_SG2260
 sgrt::sgrtStream_t TPUStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    TORCH_INTERNAL_ASSERT(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like c10::cuda::getStreamFromPool() to get a new stream.");
    return default_streams[device_index];
  } else if (st.isExt()) {
    return reinterpret_cast<sgrt::sgrtStream_t>(stream_id);
  } else {
    auto streamType = st.getStreamType();
    TORCH_INTERNAL_ASSERT(
        streamType >= 1 && streamType <= c10_tpu::max_compile_time_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
    return streams[streamType - 1][device_index][si];
  }
}

void tpuSynchronizeDevice() {
  sgrt::SgrtDeviceSynchronize();
}

TPUStream getStreamFromExternal(
    tpuRtStream_t ext_stream,
    DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return TPUStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}
#endif // BACKEND_SG2260

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
TPUStream getStreamFromPool(
    const int priority, 
    c10::DeviceIndex device_index) {
  initTPUStreamsOnce();
  if (device_index == -1)
    device_index = current_device();

  TORCH_CHECK(
    priority <= 0,
    "Expected tpu stream priority to be less than or equal to 0, got ",
    priority);
  check_tpu(device_index);
  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  auto pri_idx = -priority;
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  return TPUStreamForId(device_index, makeStreamId(StreamIdType(StreamIdType::DEFAULT), idx));
}

TPUStream getStreamFromPool(
    const bool isHighPriority,
    c10::DeviceIndex device_index) {
  initTPUStreamsOnce();
  int priority = isHighPriority ? -c10_tpu::max_compile_time_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device_index);
}

TPUStream getDefaultTPUStream(c10::DeviceIndex device_index) {
  initTPUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_tpu(device_index);
  return TPUStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

TPUStream getCurrentTPUStream(DeviceIndex device_index) {
  initTPUStreamsOnce();
  if ( device_index == -1 ) {
    device_index = current_device();
  }
  check_tpu(device_index);
  return TPUStreamForId(device_index, current_streams[device_index]);
}

void setCurrentTPUStream(TPUStream stream) {
  initTPUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const TPUStream& s) {
  return stream << s.unwrap();
}

} // namespace c10_tpu
