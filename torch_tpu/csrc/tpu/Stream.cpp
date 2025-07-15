#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include "torch_tpu/csrc/core/TPUGuard.h"
#include "torch_tpu/csrc/core/TPUFunction.h"
#include <structmember.h>

#include "torch_tpu/csrc/tpu/Stream.h"
#include "torch_tpu/csrc/tpu/Module.h"

PyObject *THPTStreamClass = nullptr;

static PyObject* THPTStream_pynew(
    PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS


  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority",
      "stream_id",
      "device_index",
      "device_type",
      "stream_ptr",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
      args,
      kwargs,
      "|iLLLK",
      const_cast<char**>(kwlist),
      &priority,
      &stream_id,
      &device_index,
      &device_type,
      &stream_ptr)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  c10_tpu::TPUStream stream =
    (stream_id || device_index || device_type) ?
    c10_tpu::TPUStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type)) :
    c10_tpu::getStreamFromPool();

  THPTStream* self = (THPTStream *)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  self->tpudnn_handle = (int64_t)((tpudnnHandle_t)stream);
  new (&self->tpu_stream) c10_tpu::TPUStream(stream);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPTStream_dealloc(THPTStream *self) {
  self->tpu_stream.~TPUStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THPTStream_get_device(THPTStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->tpu_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTStream_get_tpu_stream(THPTStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->tpu_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTStream_get_priority(THPTStream *self, void *unused) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(false, "TPU dose not support Stream.get_priority() currently.");
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTStream_priority_range() {
  HANDLE_TH_ERRORS
  TORCH_CHECK(false, "TPU does not support Stream.priority_range() currently.");
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTStream_query(THPTStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->tpu_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTStream_synchronize(THPTStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    self->tpu_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTStream_eq(THPTStream *self, THPTStream *other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->tpu_stream == other->tpu_stream);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THPTStream_members[] = {
    {(char*)"stream_id", T_ULONGLONG, offsetof(THPTStream, stream_id), READONLY, nullptr},
    {(char*)"device_type", T_ULONGLONG, offsetof(THPTStream, device_type), READONLY, nullptr},
    {(char*)"device_index", T_ULONGLONG, offsetof(THPTStream, device_index), READONLY, nullptr},
    {(char*)"tpudnn_handle", T_ULONGLONG, offsetof(THPTStream, tpudnn_handle), READONLY, nullptr},
    {nullptr}
};

static struct PyGetSetDef THPTStream_properties[] = {
    {"device", (getter)THPTStream_get_device, nullptr, nullptr, nullptr},
    {"tpu_stream", (getter)THPTStream_get_tpu_stream, nullptr, nullptr, nullptr},
    {"priority", (getter)THPTStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef THPTStream_methods[] = {
    {(char*)"query", (PyCFunction)THPTStream_query, METH_NOARGS, nullptr},
    {(char*)"synchronize", (PyCFunction)THPTStream_synchronize, METH_NOARGS, nullptr},
    {(char*)"priority_range", (PyCFunction)(void(*)(void))THPTStream_priority_range, METH_STATIC | METH_NOARGS, nullptr},
    {(char*)"__eq__", (PyCFunction)THPTStream_eq, METH_O, nullptr},
    {nullptr}
};

PyTypeObject THPTStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_tpu._C._TPUStreamBase",            /* tp_name */
  sizeof(THPTStream),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTStream_dealloc,        /* tp_dealloc */
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
  THPTStream_methods,                    /* tp_methods */
  THPTStream_members,                    /* tp_members */
  THPTStream_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPTStream_pynew,                      /* tp_new */
};


void THPTStream_init(PyObject *module)
{
  Py_INCREF(THPStreamClass);
  THPTStreamType.tp_base = THPStreamClass;
  THPTStreamClass = (PyObject*)&THPTStreamType;
  if (PyType_Ready(&THPTStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPTStreamType);
  if (PyModule_AddObject(module, "_TPUStreamBase", (PyObject *)&THPTStreamType) < 0) {
    throw python_error();
  }
}
