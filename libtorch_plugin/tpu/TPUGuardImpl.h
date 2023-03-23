#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <TPUDeviceManager.h>

namespace c10
{
namespace tpu
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

  Stream getStream ( Device ) const noexcept override
  {
    // no-op
    return Stream ( Stream::DEFAULT, Device ( DeviceType::TPU, -1 ) );
  }

  // NB: These do NOT set the current device
  Stream exchangeStream ( Stream ) const noexcept override
  {
    // no-op
    return Stream ( Stream::DEFAULT, Device ( DeviceType::TPU, -1 ) );
  }

  DeviceIndex deviceCount() const noexcept override
  {
    return ::tpu::TPUGetDeviceCount();
  }

  // Event-related functions
  void record ( void ** /*event*/,
                const Stream & /*stream*/,
                const DeviceIndex /*device_index*/,
                const EventFlag /*flag*/ ) const override
  {
    TORCH_CHECK ( false, DeviceType::TPU,
                  " backend doesn't support events." );
  }

  void block ( void * /*event*/, const Stream & /*stream*/ ) const override
  {
    TORCH_CHECK ( false, DeviceType::TPU,
                  " backend doesn't support events." )
  }

  bool queryEvent ( void * /*event*/ ) const override
  {
    TORCH_CHECK ( false, DeviceType::TPU,
                  " backend doesn't support events." )
  }

  void destroyEvent ( void * /*event*/, const DeviceIndex /*device_index*/ )
  const noexcept override {}

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
} // namespace tpu
} // namespace c10
