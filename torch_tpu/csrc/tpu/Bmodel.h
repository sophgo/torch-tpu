#pragma once
#include <Python.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "torch_tpu/csrc/aten/TPUBmodelRuntime.h"

PyMethodDef* THPTBmodel_get_methods();

struct THPTBMRT {
    PyObject_HEAD int64_t stream_id;
    int64_t device_index;
    at_tpu::modelrt::BModelRunner bmodel_runner;
};

void THPTBMRT_init(PyObject *module);