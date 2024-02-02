#pragma once

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "torch_tpu/csrc/core/TPUStream.h"

struct THPTStream : THPStream {
  c10_tpu::TPUStream tpu_stream;
};
extern PyObject *THPTStreamClass;

void THPTStream_init(PyObject *module);

inline bool THPTStream_Check(PyObject* obj) {
  return THPTStreamClass && PyObject_IsInstance(obj, THPTStreamClass);
}