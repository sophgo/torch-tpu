#include <torch/csrc/python_headers.h>

#include "torch_tpu/csrc/utils/DeviceParser.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"
#include "torch_tpu/csrc/aten/TPUGeneratorImpl.h"
#include "torch_tpu/csrc/utils/LazyInit.h"

static PyObject* THPTModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    tpu::TPUGetInstance();
  }
  auto m = THPObjectPtr(PyImport_ImportModule("torch_tpu.tpu"));
  if (!m) {
      throw python_error();
  }

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };
  auto num_tpus = tpu::TPUGetDeviceCount();
  auto default_tpu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_tpus));
  for (int i = 0; i < num_tpus; i++) {
    auto gen = at_tpu::detail::getDefaultTPUGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_tpu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_tpu_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_set_run_yet_variable_to_false_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_tpu::utils::tpu_set_run_yet_variable_to_false();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackLong(arg);
  {
    // pybind11::gil_scoped_release no_gil;
    // init device
  }
  int pre_device = tpu::TPUGetDeviceIndex();
  if (pre_device != device) {
      tpu::TPUSetDeviceIndex(pre_device);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_tpu::utils::tpu_lazy_init();
  int device = tpu::TPUGetDeviceIndex();
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(tpu::TPUGetDeviceCount());
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THPTModule_methods[] = {
    {"_tpu_init", (PyCFunction)THPTModule_initExtension, METH_NOARGS, nullptr},
    {"_tpu_set_run_yet_variable_to_false", (PyCFunction)THPTModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
    {"_tpu_setDevice", (PyCFunction)THPTModule_setDevice_wrap, METH_O, nullptr},
    {"_tpu_getDevice", (PyCFunction)THPTModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_tpu_getDeviceCount", (PyCFunction)THPTModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
    {nullptr}
};

PyMethodDef* THPTModule_get_methods() {
  return THPTModule_methods;
}