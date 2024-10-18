#pragma once
#include <torch/csrc/python_headers.h>

void RegisterTPUProperties(PyObject *module);
void BindGetDeviceProperties(PyObject *module);

PyMethodDef* THPTModule_get_methods();