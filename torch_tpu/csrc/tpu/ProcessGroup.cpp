#if defined BACKEND_SG2260

#include <pybind11/chrono.h>
#include <torch/python.h>

#include "torch_tpu/csrc/distributed/c10d/ProcessGroupSCCLHost.hpp"
#include "torch_tpu/csrc/distributed/c10d/ProcessGroupSCCL.hpp"

__attribute__((constructor)) void ProcessGroupSCCLHostConstructor() {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("SCCLHOST", py::cpp_function(c10d::ProcessGroupSCCLHost::createProcessGroupSCCLHost));
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("createProcessGroupSCCLHost", &ProcessGroupSCCLHost::createProcessGroupSCCLHost);
// }

__attribute__((constructor)) void ProcessGroupSCCLConstructor() {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend =
      module.attr("Backend").attr("register_backend");
  register_backend("SCCL", py::cpp_function(c10d::ProcessGroupSCCL::createProcessGroupSCCL), true, "tpu");
}

#endif