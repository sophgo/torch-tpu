#pragma once

#include <cstdint>
#include <utility>
#include "torch_tpu/csrc/core/TPUStream.h"
#include "torch_tpu/csrc/core/TPUGuard.h"
#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUException.h"

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
  TPUEvent() noexcept = default;  // Default value for `flags` is specified below
  TPUEvent(unsigned int flags) : flags_(flags) {}
  TPUEvent(const TPUEvent&) = delete;
  TPUEvent(TPUEvent&& other) { moveHelper(std::move(other)); }
  ~TPUEvent();

  bool  query() const;
  void  record(const TPUStream& stream);
  void  record();

  void  block(const TPUStream& stream);
  float elapsed_time(const TPUEvent& other) const;
  void  synchronize() const;

  bool  isCreated()               const { return is_created_; }
  c10::DeviceIndex device_index() const { return device_index_;}
  c10::optional<at::Device> device() const {
    if (is_created_) { return at::Device(c10::DeviceType::PrivateUse1, device_index_); }
    else { return {}; }
  }

  TPUEvent& operator=(const TPUEvent&) = delete;
  TPUEvent& operator=(TPUEvent&& other) {
    if (this != &other) { moveHelper(std::move(other)); }
    return *this;
  }
  void* event() const { return (void*)0; }
private:
  unsigned int flags_            = 0; //sgrt::SG_EVENT_DEFAULT
  bool is_created_               = false;
  bool was_recorded_             = false;
  c10::DeviceIndex device_index_ = -1;
  class Impl;
  std::shared_ptr<class Impl> event_ = nullptr;

  void createEvent(c10::DeviceIndex device_index);
  void moveHelper(TPUEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace at_tpu
