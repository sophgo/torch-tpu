#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include "TPUDeviceManager.h"
#include "torch_tpu/csrc/core/TPUStream.h"

using namespace c10;

namespace c10_tpu
{
namespace impl
{
struct TPUGuardImpl final : public c10::impl::DeviceGuardImplInterface
{

  static constexpr DeviceType static_type = DeviceType::TPU;

  TPUGuardImpl() {}

  explicit TPUGuardImpl ( DeviceType t )
  {
    TORCH_INTERNAL_ASSERT ( t == DeviceType::TPU );
  }

  DeviceType type() const override
  {
    return DeviceType::TPU;
  }

  Device exchangeDevice ( Device d ) const override
  {
    TORCH_INTERNAL_ASSERT ( d.type() == DeviceType::TPU );
    Device old_device = getDevice();
    if ( old_device.index() != d.index() )
    {
      ::tpu::TPUSetDeviceIndex ( d.index() );
    }
    return old_device;
  }

  Device getDevice() const override
  {
    int device = ::tpu::TPUGetDeviceIndex ();
    return Device ( DeviceType::TPU, device );
  }

  void setDevice ( Device d ) const override
  {
    TORCH_INTERNAL_ASSERT ( d.type() == DeviceType::TPU );
    Device current_device = getDevice();
    if ( current_device != d )
    {
      ::tpu::TPUSetDeviceIndex ( d.index() );
    }
  }

  void uncheckedSetDevice ( Device d ) const noexcept override
  {
    ::tpu::TPUSetDeviceIndex ( d.index() );
  }

  Stream getStream ( Device d ) const noexcept override
  {
    return c10_tpu::getCurrentTPUStream(d.index()).unwrap();
  }

  Stream getDefaultStream(c10::Device d) const override
  {
    return c10_tpu::getDefaultTPUStream(d.index());
  }

  // NB: These do NOT set the current device
  Stream exchangeStream ( Stream s) const noexcept override
  {
    c10_tpu::TPUStream cs(s);
    auto old_stream = c10_tpu::getCurrentTPUStream(s.device().index());
    c10_tpu::setCurrentTPUStream(cs);
    return old_stream.unwrap();
  }

  DeviceIndex deviceCount() const noexcept override
  {
    return ::tpu::TPUGetDeviceCount();
  }

  // Event-related functions
  void createEvent(void* sg_event, const c10::EventFlag flag) const
  {
    LOG(WARNING) << "flag is no use : ";
    tpuEventCreate(reinterpret_cast<tpuEvent_t*>(sg_event));
  }

  void destroyEvent ( void * event, const DeviceIndex device_index )  const noexcept override
  {
    // TODO: Implement destroyEvent for NPU events.
    // int current_device = -1;
    // TORCH_CHECK(tpuGetDevice(&current_device) == tpuSuccess);
    // TORCH_CHECK(device_index == -1 || device_index == current_device);
    // tpuEventDestroy(reinterpret_cast<tpuEvent_t*>(event));
  }


  void record ( void ** event,
                const Stream & stream,
                const DeviceIndex device_index,
                const EventFlag flag ) const override
  {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");
    auto tpu_event = static_cast<tpuEvent_t>(*event);
    c10_tpu::TPUStream tpu_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!tpu_event) {
      tpuEventCreate(&tpu_event);
    }
    tpuEventRecord(tpu_event, tpu_stream);
    // Makes the void* point to the (possibly just allocated) NPU event
    *event = tpu_event;

    // Resets device
    setDevice(orig_device);
  }

  void block ( void * event, const Stream & stream ) const override
  {
    if (!event)
      return;
    auto tpu_event = static_cast<tpuEvent_t>(event);
    c10_tpu::TPUStream tpu_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    tpuStreamWaitEvent(tpu_stream, tpu_event);
    setDevice(orig_device);
  }

  bool queryEvent ( void * event ) const override
  {
    // TODO: Implement queryEvent for NPU events.
    // Currently, we return true to indicate that the event has been recorded.
    return true;
  }


#if 0
  // Stream-related functions
  bool queryStream ( const Stream & /*stream*/ ) const override
  {
    return true;
  }

  void synchronizeStream ( const Stream & /*stream*/ ) const override
  {
    // Don't wait for anything.
  }
#endif
};
} // namespace impl
} // namespace c10_tpu
