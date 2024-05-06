#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include "TPUDeviceManager.h"

#ifdef BACKEND_SG2260
#include "torch_tpu/csrc/core/Interface/sgrtInterface.h"
#include "torch_tpu/csrc/core/TPUStream.h"
using namespace c10_tpu::sgrt;
#endif

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
  #ifdef BACKEND_SG2260
    return c10_tpu::getCurrentTPUStream(d.index()).unwrap();
  #else
    return Stream ( Stream::DEFAULT, Device ( DeviceType::TPU, -1 ) );    // no-op
  #endif
  }

  Stream getDefaultStream(c10::Device d) const override
  {
  #ifdef BACKEND_SG2260
    return c10_tpu::getDefaultTPUStream(d.index());
  #else
    TORCH_CHECK(false, "Backend doesn't support acquiring a default stream.")
  #endif
  }

  // NB: These do NOT set the current device
  Stream exchangeStream ( Stream s) const noexcept override
  {
  #ifdef BACKEND_SG2260
  c10_tpu::TPUStream cs(s);
  auto old_stream = c10_tpu::getCurrentTPUStream(s.device().index());
  c10_tpu::setCurrentTPUStream(cs);
  return old_stream.unwrap();
  #else
    return Stream ( Stream::DEFAULT, Device ( DeviceType::TPU, -1 ) ); // no-op
  #endif
  }

  DeviceIndex deviceCount() const noexcept override
  {
    return ::tpu::TPUGetDeviceCount();
  }

  // Event-related functions
  void createEvent(void* sg_event, const c10::EventFlag flag) const 
  {
  #ifdef BACKEND_SG2260
    SgrtCreateEventWithFlag((sgrtEvent_t*)sg_event, (uint32_t)flag);
  #else
  #endif
  }

  void destroyEvent ( void * event, const DeviceIndex device_index )  const noexcept override 
  {
  #ifdef BACKEND_SG2260
    if (!event) return;
  #else
  #endif
  }


  void record ( void ** event,
                const Stream & stream,
                const DeviceIndex device_index,
                const EventFlag flag ) const override
  {
  #ifdef BACKEND_SG2260
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");
    sgrtEvent_t tpu_event = static_cast<sgrtEvent_t>(*event);
    c10_tpu::TPUStream tpu_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!tpu_event) {
      SgrtCreateEvent(&tpu_event);
    }
    SgrtEventRecord(tpu_event, tpu_stream);
    // Makes the void* point to the (possibly just allocated) NPU event
    *event = tpu_event;

    // Resets device
    setDevice(orig_device);
  #else
  #endif
  }

  void block ( void * event, const Stream & stream ) const override
  {
  #ifdef BACKEND_SG2260
    if (!event)
      return;
    sgrtEvent_t tpu_event = static_cast<sgrtEvent_t>(event);
    c10_tpu::TPUStream tpu_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    SgrtStreamWaitEvent(tpu_stream, tpu_event);
    setDevice(orig_device);
  #else
  #endif
  }

  bool queryEvent ( void * event ) const override
  {
  #ifdef BACKEND_SG2260
    if (!event)
      return true;
    sgrtEvent_t tpu_event = static_cast<sgrtEvent_t>(event);
    sgrtEventRecordedStatus status =
        SG_EVENT_RECORDED_STATUS_NOT_READY;
    SgrtQueryEventStatus(tpu_event, &status);
    return (status == SG_EVENT_RECORDED_STATUS_COMPLETE);
  #else
    return true;
  #endif
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
