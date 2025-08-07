#include <torch/csrc/python_headers.h>
#include <pybind11/chrono.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_tpu/csrc/aten/TPUGraph.h"
#include "tpu_runtime_api.h"


template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THPTGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at_tpu::graph_pool_handle);

  shared_ptr_class_<::at_tpu::TPUGraph>(torch_C_m, "_TPUGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::at_tpu::TPUGraph& self,
             std::optional<c10_tpu::MempoolId_t> pool_opt) {
            c10_tpu::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10_tpu::MempoolId_t{0, 0};
            return self.capture_begin(pool);
          },
          py::arg("pool"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at_tpu::TPUGraph::capture_end))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(&at_tpu::TPUGraph::instantiate))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at_tpu::TPUGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at_tpu::TPUGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at_tpu::TPUGraph::pool))
      .def(
          "raw_tpu_graph",
          [](::at_tpu::TPUGraph& self) {
            auto graph = self.raw_tpu_graph();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of TPUGraph_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>());
}
