#pragma once

#include <string>

#include <c10/util/Registry.h>
#include <sophon/config.h>
#include <sophon/transport/device.h>

namespace c10d {

class TORCH_API SCCLDeviceFactory {
 public:
  // Create new device instance for specific interface.
  static std::shared_ptr<::sophon::transport::Device> makeDeviceForInterface(
      const std::string& interface);

  // Create new device instance for specific hostname or address.
  static std::shared_ptr<::sophon::transport::Device> makeDeviceForHostname(
      const std::string& hostname);
};

C10_DECLARE_SHARED_REGISTRY(
    SophonDeviceRegistry,
    ::sophon::transport::Device,
    const std::string&, /* interface */
    const std::string& /* hostname */);

} // namespace c10d