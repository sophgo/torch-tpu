#include <stdlib.h>

#include <c10/util/Exception.h>

#include <gloo/transport/tcp/device.h>


#include "SCCLHostDeviceFactory.hpp"

namespace c10d {

C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING(
    SCCLDeviceRegistry,
    ::gloo::transport::Device,
    const std::string& /* interface */,
    const std::string& /* hostname */);

static std::shared_ptr<::gloo::transport::Device> makeTCPDevice(
    const std::string& interfaceName,
    const std::string& hostname) {
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "SCCLHostDeviceFactory::makeTCPDevice(): interface or hostname "
      "can't be empty");

  ::gloo::transport::tcp::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  return ::gloo::transport::tcp::CreateDevice(attr);
}

// Registry priority is per key identifier. We register TCP to `LINUX` for
// the flexibility of other application to override by priority. Register
// TCP to `TCP` for env "GLOO_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(SCCLDeviceRegistry, LINUX, makeTCPDevice);
C10_REGISTER_CREATOR(SCCLDeviceRegistry, TCP, makeTCPDevice);


namespace {
std::shared_ptr<::gloo::transport::Device> makeGlooDevice(
    const std::string& interfaceName,
    const std::string& hostName) {
  static auto transportName = getenv("GLOO_DEVICE_TRANSPORT");
  if (transportName) {
    return SCCLDeviceRegistry()->Create(transportName, interfaceName, hostName);
  }

#ifdef __linux__
  return SCCLDeviceRegistry()->Create("LINUX", interfaceName, hostName);
#endif

  return nullptr;
}
} // anonymous namespace

std::shared_ptr<::gloo::transport::Device> SCCLHostDeviceFactory::
    makeDeviceForInterface(const std::string& interfaceName) {
  auto device = makeGlooDevice(interfaceName, "");
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForInterface(): unsupported gloo device");
  }
  return device;
}

std::shared_ptr<::gloo::transport::Device> SCCLHostDeviceFactory::
    makeDeviceForHostname(const std::string& hostname) {
  auto device = makeGlooDevice("", hostname);
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForHostname(): unsupported gloo device");
  }
  return device;
}

} // namespace c10d
