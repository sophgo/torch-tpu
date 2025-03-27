#ifndef THPT_EVENT_INC
#define THPT_EVENT_INC

#include "torch_tpu/csrc/aten/TPUEvent.h"
#include <torch/csrc/python_headers.h>

struct THPTEvent {
  PyObject_HEAD
  at_tpu::TPUEvent tpu_event;
};

void THPTEvent_init(PyObject *module);

// extern PyObject *THPTEventClass;
// inline bool THPTEvent_Check(PyObject* obj) {
//   return THPTEventClass && PyObject_IsInstance(obj, THPTEventClass);
// }

#endif // THPT_EVENT_INC
