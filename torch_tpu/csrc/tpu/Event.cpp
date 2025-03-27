#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <structmember.h>

#include "torch_tpu/csrc/core/TPULog.h"
#if defined BACKEND_SG2260
#include "torch_tpu/csrc/tpu/Event.h"
#include "torch_tpu/csrc/tpu/Stream.h"

PyObject *THPTEventClass = nullptr;

static PyObject* THPTEvent_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  char *kwlist[] = {(char*)"enable_timing", (char*)"blocking", (char*)"interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbb", kwlist,
      &enable_timing, &blocking, &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THPTEvent* self = (THPTEvent *)ptr.get();

  unsigned int flags = enable_timing ? 0 : 1;
  new (&self->tpu_event) at_tpu::TPUEvent(flags);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPTEvent_dealloc(THPTEvent *self) {
  self->tpu_event.~TPUEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THPTEvent_get_tpu_event(THPTEvent *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->tpu_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTEvent_get_device(THPTEvent *self, void *unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->tpu_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTEvent_record(THPTEvent *self, THPTStream *stream) {
  HANDLE_TH_ERRORS
  self->tpu_event.record(stream->tpu_stream);
  SOPHON_LOGI("Event: record api is successfully executed.");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTEvent_wait(THPTEvent *self, THPTStream *stream) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    self->tpu_event.block(stream->tpu_stream);
    SOPHON_LOGI("Event: wait api is successfully executed.");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTEvent_query(THPTEvent *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->tpu_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTEvent_elapsed_time(THPTEvent *self, THPTEvent *other) {
  HANDLE_TH_ERRORS
  return PyFloat_FromDouble(self->tpu_event.elapsed_time(other->tpu_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTEvent_synchronize(THPTEvent *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    self->tpu_event.synchronize();
    SOPHON_LOGI("Event: synchronize api is successfully executed.");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THPTEvent_properties[] = {
    {"device", (getter)THPTEvent_get_device, nullptr, nullptr, nullptr},
    {"tpu_event", (getter)THPTEvent_get_tpu_event, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef THPTEvent_methods[] = {
    {(char*)"record", (PyCFunction)THPTEvent_record, METH_O, nullptr},
    {(char*)"wait", (PyCFunction)THPTEvent_wait, METH_O, nullptr},
    {(char*)"query", (PyCFunction)THPTEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", (PyCFunction)THPTEvent_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", (PyCFunction)THPTEvent_synchronize, METH_NOARGS, nullptr},
    {nullptr}
};

PyTypeObject THPTEventType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_tpu._C._TPUEventBase",          /* tp_name */
  sizeof(THPTEvent),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTEvent_dealloc,         /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPTEvent_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  THPTEvent_properties,                  /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPTEvent_pynew,                       /* tp_new */
};
#endif // BACKEND_SG2260

void THPTEvent_init(PyObject *module) {
#ifdef BACKEND_SG2260
  THPTEventClass = (PyObject*)&THPTEventType;
  if (PyType_Ready(&THPTEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPTEventType);
  if (PyModule_AddObject(module, "_TPUEventBase", (PyObject *)&THPTEventType) < 0) {
    throw python_error();
  }
#endif
}
