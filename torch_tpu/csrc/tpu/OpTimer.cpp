#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include "torch_tpu/csrc/core/TPUCtypeApi.h"
#include "torch_tpu/csrc/utils/LazyInit.h"

PyObject* THPTOpTimer_opreset_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_op_timer_reset();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTOpTimer_opdump_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_op_timer_dump();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTOpTimer_globalreset_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_timer_reset();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTOpTimer_globaldump_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_timer_dump();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THPTOpTimer_methods[] = {
    {"_Timer_OpReset", (PyCFunction)THPTOpTimer_opreset_wrap, METH_NOARGS, nullptr},
    {"_Timer_OpDump", (PyCFunction)THPTOpTimer_opdump_wrap, METH_NOARGS, nullptr},
    {"_Timer_GlobalReset", (PyCFunction)THPTOpTimer_globalreset_wrap, METH_NOARGS, nullptr},
    {"_Timer_GlobalDump", (PyCFunction)THPTOpTimer_globaldump_wrap, METH_NOARGS, nullptr},
    {nullptr}
};

PyMethodDef* THPTOpTimer_get_methods() {
  return THPTOpTimer_methods;
}