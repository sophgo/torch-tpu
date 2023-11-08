#pragma once
#include <torch/csrc/python_headers.h>
#include<ATen/Device.h>

struct TORCH_API THPTDevice{
    PyObject_HEAD
    at::Device device;
};

TORCH_API extern PyTypeObject THPTDeviceType;

inline bool THPTDevice_Check(PyObject *obj) {
    return Py_TYPE(obj) == &THPTDeviceType;
}

PyObject* THPTDevice_New(const at::Device& device);

void THPTDevice_init(PyObject * module);