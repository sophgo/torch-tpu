#pragma once

#include "torch_tpu/csrc/core/TPUStream.h"
#include "torch_tpu/csrc/core/TPUGuard.h"
#include "torch_tpu/csrc/core/Interface/sgrtInterface.h"
#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUException.h"

#include <cstdint>
#include <utility>

using namespace c10_tpu;

namespace at_tpu {
/*
* TPUEvents are movable not copyable wrappers around TPU's events.
*
* TPUEvents are constructed lazily when first recorded unless it is
* reconstructed from a cudaIpcEventHandle_t. The event has a device, and this
* device is acquired from the first recording stream. However, if reconstructed
* from a handle, the device should be explicitly specified; or if ipc_handle() is
* called before the event is ever recorded, it will use the current device.
* Later streams that record the event must match this device.
*/
struct TPUEvent {
  // Constructors
  // Default value for `flags` is specified below
  TPUEvent() noexcept = default;
  TPUEvent(unsigned int flags) : flags_(flags) {}

  // tpu do not support IpcEventHandle until now
  // CUDAEvent(
  //     DeviceIndex device_index, const cudaIpcEventHandle_t* handle) {
  //     device_index_ = device_index;
  //     CUDAGuard guard(device_index_);

  //     AT_CUDA_CHECK(cudaIpcOpenEventHandle(&event_, *handle));
  //     is_created_ = true;
  // }

  ~TPUEvent() {
    try {
      if (is_created_){
        TPUGuard guard(device_index_);
        sgrt::SgrtEventDestroy(event_);
      }
    }
    catch (...) {
      // stay consistent with pytorch, no throw
    }
  }

  TPUEvent(const TPUEvent&) = delete;
  TPUEvent& operator=(const TPUEvent&) = delete;

  TPUEvent(TPUEvent&& other) { moveHelper(std::move(other)); }
  TPUEvent& operator=(TPUEvent&& other) {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator sgrt::sgrtEvent_t() const { return event(); }

  c10::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(c10::DeviceType::PrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  c10::DeviceIndex device_index() const {return device_index_;}
  sgrt::sgrtEvent_t event() const { return event_; }

  bool query() const {
    if (!is_created_) {
      return true;
    }

    sgrt::sgrtEventRecordedStatus currStatus =
        sgrt::SG_EVENT_RECORDED_STATUS_NOT_READY;
    C10_TPU_CHECK(sgrt::SgrtQueryEventStatus(event_, &currStatus));

    if (currStatus == sgrt::SG_EVENT_RECORDED_STATUS_COMPLETE) {
      return true;
    }
    return false;
  }

  void record() { record(getCurrentTPUStream()); }

  void recordOnce(const TPUStream& stream) {
    if (!was_recorded_) record(stream);
  }

  void record(const TPUStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
        " does not match recording stream's device ", stream.device_index(), ".");
    TPUGuard guard(device_index_);
    sgrt::SgrtEventRecord(event_, stream.stream());
    was_recorded_ = true;
  }

  void block(const TPUStream& stream) {
    if (is_created_) {
      TPUGuard guard(stream.device_index());
      sgrt::SgrtStreamWaitEvent(stream.stream(), event_);
    }
  }

  float elapsed_time(const TPUEvent& other) const {
    TORCH_CHECK(is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    C10_TPU_CHECK(sgrt::SgrtEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      C10_TPU_CHECK(sgrt::SgrtSynchronizeEvent(event_));
      SOPHON_LOGI("Event: sgrtSynchronizeEvent is successfully executed.");
    }
  }

  // tpu do not support IpcEventHandle until now

private:
  unsigned int flags_ = 0; //sgrt::SG_EVENT_DEFAULT
  bool is_created_ = false;
  bool was_recorded_ = false;
  c10::DeviceIndex device_index_ = -1;
  sgrt::sgrtEvent_t event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    TPUGuard guard(device_index_);
    C10_TPU_CHECK(sgrt::SgrtCreateEventWithFlag(&event_, flags_));
    SOPHON_LOGI("Event: SgrtCreateEvent is successfully executed.");
    is_created_ = true;
  }

  void moveHelper(TPUEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace at_tpu
