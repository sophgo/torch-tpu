#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_tpu/csrc/utils/LazyInit.h"

namespace torch_tpu {
namespace utils {

static bool tpu_run_yet = false;

void tpu_lazy_init() {
  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (!tpu_run_yet) {
    auto module = THPObjectPtr(PyImport_ImportModule("torch_tpu.tpu"));
    if (!module) {
        throw python_error();
    }
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) {
        throw python_error();
    }
    tpu_run_yet = true;
  }
}

void tpu_set_run_yet_variable_to_false() {
  tpu_run_yet = false;
}



}
}