#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include "torch_tpu/csrc/core/TPUCtypeApi.h"
#include "torch_tpu/csrc/ops/my_ops/ops.hpp"
#include "torch_tpu/csrc/utils/LazyInit.h"

PyObject* THPTOpTimer_opreset_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_op_timer_reset();
  at::reset_tpudnn_optimer();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTOpTimer_opdump_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_op_timer_dump();
  at::dump_tpudnn_optimer();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTOpTimer_oppause_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_op_timer_pause();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTOpTimer_opstart_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  tpu_op_timer_start();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THPTOpTimer_methods[] = {
    {"_Timer_OpReset", (PyCFunction)THPTOpTimer_opreset_wrap, METH_NOARGS, nullptr},
    {"_Timer_OpDump", (PyCFunction)THPTOpTimer_opdump_wrap, METH_NOARGS, nullptr},
    {"_Timer_OpPause", (PyCFunction)THPTOpTimer_oppause_wrap, METH_NOARGS, nullptr},
    {"_Timer_OpStart", (PyCFunction)THPTOpTimer_opstart_wrap, METH_NOARGS, nullptr},
    {nullptr}
};

PyMethodDef* THPTUtils_get_methods() {
  return THPTOpTimer_methods;
}