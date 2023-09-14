#include <stdlib.h>

#include <c10/util/Exception.h>

#include <sophon/transport/tcp/device.h>

#include "SophonDeviceFactory.hpp"

namespace c10d {

C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING(SophonDeviceRegistry,
                                           ::sophon::transport::Device,
                                           const std::string & /* interface */,
                                           const std::string & /* hostname */);

static std::shared_ptr<::sophon::transport::Device>
makeTCPDevice(const std::string &interfaceName, const std::string &hostname) {
  TORCH_CHECK(!interfaceName.empty() || !hostname.empty(),
              "SophonDeviceFactory::makeTCPDevice(): interface or hostname "
              "can't be empty");

  ::sophon::transport::tcp::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  return ::sophon::transport::tcp::CreateDevice(attr);
}

// Registry priority is per key identifier. We register TCP to `LINUX` for
// the flexibility of other application to override by priority. Register
// TCP to `TCP` for env "SOPHON_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(SophonDeviceRegistry, LINUX, makeTCPDevice);
C10_REGISTER_CREATOR(SophonDeviceRegistry, TCP, makeTCPDevice);

namespace {
std::shared_ptr<::sophon::transport::Device>
makeSophonDevice(const std::string &interfaceName,
                 const std::string &hostName) {
  static auto transportName = getenv("SOPHON_DEVICE_TRANSPORT");
  if (transportName) {
    return SophonDeviceRegistry()->Create(transportName, interfaceName,
                                          hostName);
  }

#ifdef __linux__
  return SophonDeviceRegistry()->Create("LINUX", interfaceName, hostName);
#endif

  return nullptr;
}
} // anonymous namespace

std::shared_ptr<::sophon::transport::Device>
SophonDeviceFactory::makeDeviceForInterface(const std::string &interfaceName) {
  auto device = makeSophonDevice(interfaceName, "");
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForInterface(): unsupported sophon device");
  }
  return device;
}

std::shared_ptr<::sophon::transport::Device>
SophonDeviceFactory::makeDeviceForHostname(const std::string &hostname) {
  auto device = makeSophonDevice("", hostname);
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForHostname(): unsupported sophon device");
  }
  return device;
}

} // namespace c10d
