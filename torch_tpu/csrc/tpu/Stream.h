#pragma once

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "torch_tpu/csrc/core/TPUStream.h"

struct THPTStream : THPStream {
    int64_t            tpudnn_handle;
    c10_tpu::TPUStream tpu_stream;
};

void THPTStream_init(PyObject *module);

// extern PyObject *THPTStreamClass;
// inline bool THPTStream_Check(PyObject* obj) {
//   return THPTStreamClass && PyObject_IsInstance(obj, THPTStreamClass);
// }