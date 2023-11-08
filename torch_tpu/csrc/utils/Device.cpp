#include <sstream>

#include <c10/util/Exception.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_tpu/csrc/utils/Device.h"
#include "torch_tpu/csrc/utils/DeviceParser.h"

PyObject *THPTDevice_repr(THPTDevice* self){
  std::ostringstream oss;
  if (self->device.type() == at_tpu::key::NativeDeviceType) {
    oss << "device(type=\'" << at_tpu::key::tpu_device_str << "\'";
  } else {
    oss << "device(type=\'" << self->device.type() << "\'";
  }
  
  if (self->device.has_index()) {
    // `self->device.index()` returns uint8_t which is treated as ascii while printing,
    // hence casting it to uint16_t.
    oss << ", index=" << static_cast<uint16_t>(self->device.index());
  }
  oss << ")";
  return THPUtils_packString(oss.str().c_str());
}

static Py_ssize_t THPTDevice_hash(THPTDevice *self){
  HANDLE_TH_ERRORS
  return static_cast<Py_ssize_t>(static_cast<Py_ssize_t>(std::hash<at::Device>{}(self->device)) % std::numeric_limits<Py_ssize_t>::max());
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPTDevice_str(THPTDevice *self)
{
  std::ostringstream oss;
  std::string str = c10::DeviceTypeName(self->device.type(), true);
  if (at_tpu::key::default_device_str == str) {
      str = at_tpu::key::tpu_device_str;
  }
  if (self->device.has_index()) {
      str.push_back(':');
      str.append(std::to_string(self->device.index()));
  }
  oss << str;
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPTDevice_rc(PyObject *a, PyObject *b, int op) {
    HANDLE_TH_ERRORS
    if (!THPTDevice_Check(a) || !THPTDevice_Check(b)) {
        // Py_RETURN_NOTIMPLEMENTED not in python 2.
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    THPTDevice *da = reinterpret_cast<THPTDevice*>(a);
    THPTDevice *db = reinterpret_cast<THPTDevice*>(b);

    switch (op) {
        case Py_EQ:
            if (da->device == db->device) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        case Py_NE:
            if (da->device == db->device) {
                Py_RETURN_FALSE;
            } else {
                Py_RETURN_TRUE;
            }
        case Py_LT:
        case Py_LE:
        case Py_GT:
        case Py_GE:
            throw torch::TypeError("comparison not implemented");
        default:
            throw torch::TypeError("unexpected comparison op");
    }
    END_HANDLE_TH_ERRORS
}

PyObject *THPTDevice_type(THPTDevice *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  std::ostringstream oss;
  if (self->device.type() == at_tpu::key::NativeDeviceType) {
    oss << at_tpu::key::tpu_device_str;
  } else {
    oss << self->device.type();
  }
  return THPUtils_packString(oss.str().c_str());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPTDevice_index(THPTDevice *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  if (self->device.has_index()) {
    return THPUtils_packInt64(self->device.index());
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPTDevice_reduce(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPTDevice*)_self;
  auto ret = THPObjectPtr{PyTuple_New(2)};
  if (!ret) {
      throw python_error();
  }

  py::object torch_module = py::module::import("torch_tpu._C");
  py::object torch_device = torch_module.attr("device");
  PyTuple_SET_ITEM(ret.get(), 0, torch_device.release().ptr());

  THPObjectPtr args;
  std::ostringstream oss;
  if (self->device.type() == at_tpu::key::NativeDeviceType) {
    oss << at_tpu::key::tpu_device_str;
  } else {
    oss << self->device.type();
  }
  if (self->device.has_index()) {
    args = THPObjectPtr{Py_BuildValue("(si)", oss.str().c_str(), self->device.index())};
  } else {
    args = THPObjectPtr{Py_BuildValue("(s)", oss.str().c_str())};
  }
  if (!args) {
      throw python_error();
  }
  PyTuple_SET_ITEM(ret.get(), 1, args.release());

  return ret.release();
  END_HANDLE_TH_ERRORS
}

PyObject *THPTDevice_New(const at::Device& device)
{
  auto type = (PyTypeObject*)&THPTDeviceType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) {
    throw python_error();
  }
  auto self_ = reinterpret_cast<THPTDevice*>(self.get());
  self_->device = device;
  return self.release();
}

PyObject *THPTDevice_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Device(PyObject* device)",
    "Device(std::string type, int64_t? index=-1)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto as_device = at_tpu::key::parse_tpu_device(r.pyobject(0));
    return THPTDevice_New(as_device);
  } else {
    auto as_device = at_tpu::key::parse_tpu_device(r.args[0]);  // this works, because device can take strings
    auto device_type = r.string(0);
    if (as_device.has_index() && !r.isNone(1)) {
      throw std::runtime_error("type (string) must not include an index because index "
                                "was passed explicitly: " + device_type);
    }
    int32_t device_index = as_device.has_index() ? as_device.index() : -1;
    if (!r.isNone(1)) {
      device_index = r.toInt64(1);
      // -1 is allowed in ATen/C++, to mean the default device, but not in
      // Python.
      TORCH_CHECK(device_index >= 0, "Device index must not be negative");
    }
    at::Device device(as_device.type(), device_index);
    return THPTDevice_New(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

using getter = PyObject* (*)(PyObject *, void *);

static struct PyGetSetDef THPTDevice_properties[] = {
    {"type",       (getter)THPTDevice_type, nullptr, nullptr, nullptr},
    {"index",      (getter)THPTDevice_index, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef THPTDevice_methods[] = {
    {"__reduce__", THPTDevice_reduce, METH_NOARGS, nullptr},
    {nullptr}  /* Sentinel */
};

PyTypeObject THPTDeviceType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_tpu._C.device",                        /* tp_name */
  sizeof(THPTDevice),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  nullptr,                               /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  nullptr,                               /* tp_getattr */
  nullptr,                               /* tp_setattr */
  nullptr,                               /* tp_reserved */
  (reprfunc)THPTDevice_repr,              /* tp_repr */
  nullptr,                               /* tp_as_number */
  nullptr,                               /* tp_as_sequence */
  nullptr,                               /* tp_as_mapping */
  (hashfunc)THPTDevice_hash,              /* tp_hash  */
  nullptr,                               /* tp_call */
  (reprfunc)THPTDevice_str,               /* tp_str */
  nullptr,                               /* tp_getattro */
  nullptr,                               /* tp_setattro */
  nullptr,                               /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  nullptr,                               /* tp_traverse */
  nullptr,                               /* tp_clear */
  (richcmpfunc)THPTDevice_rc,             /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                               /* tp_iter */
  nullptr,                               /* tp_iternext */
  THPTDevice_methods,                     /* tp_methods */
  nullptr,                               /* tp_members */
  THPTDevice_properties,                  /* tp_getset */
  nullptr,                               /* tp_base */
  nullptr,                               /* tp_dict */
  nullptr,                               /* tp_descr_get */
  nullptr,                               /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                               /* tp_init */
  nullptr,                               /* tp_alloc */
  THPTDevice_pynew,                       /* tp_new */
};

void THPTDevice_init(PyObject *module)
{
  if (PyType_Ready(&THPTDeviceType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPTDeviceType);
  if (PyModule_AddObject(module, "device", (PyObject *)&THPTDeviceType) != 0) {
    throw python_error();
  }
}