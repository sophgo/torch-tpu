#pragma once
#include <torch/csrc/python_headers.h>

void RegisterTPUDeviceProperties(PyObject *module);
void BindGetDeviceProperties(PyObject *module);

PyMethodDef* THPTModule_get_methods();