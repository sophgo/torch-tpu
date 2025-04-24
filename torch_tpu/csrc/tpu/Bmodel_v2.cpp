#include "Bmodel.h"

#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/extension.h>
#include <torch/csrc/python_headers.h>
#include <torch/autograd.h>

#include "torch_tpu/csrc/core/TPUStream.h"
#include "torch_tpu/csrc/core/TPUCtypeApi.h"
#include "torch_tpu/csrc/utils/LazyInit.h"
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"

PyObject* THPTBMRT_InPlaceTensor(THPTBMRT* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto in_out = self->bmodel_runner.GenInplaceTensor();
  auto in = std::get<0>(in_out);
  auto out= std::get<1>(in_out);
  PyObject* py_list1 = PyList_New(in.size());
  PyObject* py_list2 = PyList_New(out.size());
  
  if (!py_list1 || !py_list2) {
    Py_XDECREF(py_list1);
    Py_XDECREF(py_list2);
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python lists");
    return NULL;
  }

  // 3. 将 tensor 转换为 Python 对象并放入 list
  for (size_t i = 0; i < in.size(); ++i) {
    PyObject* py_tensor = THPVariable_Wrap(in[i]);
    if (!py_tensor) {
      Py_DECREF(py_list1);
      Py_DECREF(py_list2);
      PyErr_SetString(PyExc_RuntimeError, "Failed to wrap tensor");
      return NULL;
    }
    PyList_SET_ITEM(py_list1, i, py_tensor);  // 偷引用
  }

  for (size_t i = 0; i < out.size(); ++i) {
    PyObject* py_tensor = THPVariable_Wrap(out[i]);
    if (!py_tensor) {
      Py_DECREF(py_list1);
      Py_DECREF(py_list2);
      PyErr_SetString(PyExc_RuntimeError, "Failed to wrap tensor");
      return NULL;
    }
    PyList_SET_ITEM(py_list2, i, py_tensor);  // 偷引用
  }

  // 4. 创建返回的 tuple (list1, list2)
  PyObject* result = PyTuple_Pack(2, py_list1, py_list2);
  Py_DECREF(py_list1);
  Py_DECREF(py_list2);
  
  if (!result) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create result tuple");
    return NULL;
  }

  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTBMRT_GetIONames(THPTBMRT* self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto in_out = self->bmodel_runner.GetIONames();
  auto in = std::get<0>(in_out);
  auto out= std::get<1>(in_out);
  PyObject* py_list1 = PyList_New(in.size());
  PyObject* py_list2 = PyList_New(out.size());
  
  if (!py_list1 || !py_list2) {
    Py_XDECREF(py_list1);
    Py_XDECREF(py_list2);
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Python lists");
    return NULL;
  }

  // 3. 将 tensor 转换为 Python 对象并放入 list
  for (size_t i = 0; i < in.size(); ++i) {
    PyObject* py_tensor = PyUnicode_FromString(in[i].c_str());
    if (!py_tensor) {
      Py_DECREF(py_list1);
      Py_DECREF(py_list2);
      PyErr_SetString(PyExc_RuntimeError, "Failed to wrap tensor");
      return NULL;
    }
    PyList_SET_ITEM(py_list1, i, py_tensor);  // 偷引用
  }

  for (size_t i = 0; i < out.size(); ++i) {
    PyObject* py_tensor = PyUnicode_FromString(out[i].c_str());
    if (!py_tensor) {
      Py_DECREF(py_list1);
      Py_DECREF(py_list2);
      PyErr_SetString(PyExc_RuntimeError, "Failed to wrap tensor");
      return NULL;
    }
    PyList_SET_ITEM(py_list2, i, py_tensor);  // 偷引用
  }

  // 4. 创建返回的 tuple (list1, list2)
  PyObject* result = PyTuple_Pack(2, py_list1, py_list2);
  Py_DECREF(py_list1);
  Py_DECREF(py_list2);
  
  if (!result) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create result tuple");
    return NULL;
  }

  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTBMRT_forward(THPTBMRT* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* pytensor_input;
  PyObject* pytensor_output;
  int non_blocking_;
  if (!PyArg_ParseTuple(args, "OOp", &pytensor_input, &pytensor_output, &non_blocking_)) {
    return NULL;
  }
  bool non_blocking = static_cast<bool>(non_blocking_);

  if (THPVariable_Check(pytensor_input) && THPVariable_Check(pytensor_output))
  {
    auto input_tensor  = THPVariable_Unpack(pytensor_input);
    auto output_tensor = THPVariable_Unpack(pytensor_output);
    self->bmodel_runner.forward(input_tensor, output_tensor, non_blocking);
  }
  else if (THPVariable_Check(pytensor_input) && PyList_Check(pytensor_output))
  {
    auto input_tensor  = THPVariable_Unpack(pytensor_input);
    std::vector<torch::Tensor> tensor_list;
    Py_ssize_t list_size = PyList_Size(pytensor_output);
    tensor_list.reserve(list_size);
    for (Py_ssize_t i = 0; i < list_size; ++i) {
      PyObject* item = PyList_GetItem(pytensor_output, i);      
      if (THPVariable_Check(item)) {
        tensor_list.push_back(THPVariable_Unpack(item));
      } else { 
        PyErr_SetString(PyExc_TypeError, "Expected a Tensor or list as second argument");
        Py_RETURN_NONE;
      }
    }
    at::TensorList tensor_(tensor_list);
    self->bmodel_runner.forward(input_tensor, tensor_, non_blocking);
  }
  else if (THPVariable_Check(pytensor_output) && PyList_Check(pytensor_input))
  {
    auto output_tensor  = THPVariable_Unpack(pytensor_output);
    std::vector<torch::Tensor> tensor_list;
    Py_ssize_t list_size = PyList_Size(pytensor_input);
    tensor_list.reserve(list_size);
    for (Py_ssize_t i = 0; i < list_size; ++i) {
      PyObject* item = PyList_GetItem(pytensor_input, i);      
      if (THPVariable_Check(item)) {
        tensor_list.push_back(THPVariable_Unpack(item));
      } else { 
        PyErr_SetString(PyExc_TypeError, "Expected a Tensor or list as second argument");
        Py_RETURN_NONE;
      }
    }
    at::TensorList tensor_(tensor_list);
    self->bmodel_runner.forward(tensor_, output_tensor, non_blocking);   
  }
  else if (PyList_Check(pytensor_input) && PyList_Check(pytensor_output))
  {
    std::vector<torch::Tensor> tensor_list_in;
    Py_ssize_t list_size = PyList_Size(pytensor_input);
    tensor_list_in.reserve(list_size);
    for (Py_ssize_t i = 0; i < list_size; ++i) {
      PyObject* item = PyList_GetItem(pytensor_input, i);      
      if (THPVariable_Check(item)) {
        tensor_list_in.push_back(THPVariable_Unpack(item));
      } else { 
        PyErr_SetString(PyExc_TypeError, "Expected a Tensor or list as second argument");
        Py_RETURN_NONE;
      }
    }
    at::TensorList tensor_in(tensor_list_in);
    std::vector<torch::Tensor> tensor_list_out;
    list_size = PyList_Size(pytensor_output);
    tensor_list_out.reserve(list_size);
    for (Py_ssize_t i = 0; i < list_size; ++i) {
      PyObject* item = PyList_GetItem(pytensor_output, i);      
      if (THPVariable_Check(item)) {
        tensor_list_out.push_back(THPVariable_Unpack(item));
      } else { 
        PyErr_SetString(PyExc_TypeError, "Expected a Tensor or list as second argument");
        Py_RETURN_NONE;
      }
    }
    at::TensorList tensor_out(tensor_list_out);
    self->bmodel_runner.forward(tensor_in, tensor_out, non_blocking);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTBMRT_setRuningNet(THPTBMRT* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const char* net_name;
  if (!PyArg_ParseTuple(args, "s", &net_name)) {
    return NULL;
  }
  self->bmodel_runner.set_running_net(net_name);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPTBMRT_pynew(
    PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS

  int64_t stream_id = 0;
  int64_t device_index = 0;
  const char* model_file;
  const char* decrypt_lib;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "bmodel_path", "decrypt_lib",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
      args,
      kwargs,
      "|LLss",
      const_cast<char**>(kwlist),
      &stream_id, &device_index, &model_file, &decrypt_lib)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THPTBMRT* self = (THPTBMRT *)ptr.get();
  self->stream_id    = stream_id;
  self->device_index = device_index;
  new (&self->bmodel_runner) at_tpu::modelrt::BModelRunner(model_file, device_index, decrypt_lib);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPTBMRT_dealloc(THPTBMRT *self) {
  self->bmodel_runner.~BModelRunner();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// static struct PyMemberDef THPTBMRT_members[] = {
//   // {(char*)"Model", T_ULONGLONG, offsetof(THPTBMRT, Model), READONLY, nullptr},
//   {nullptr},
// };

// static struct PyGetSetDef THPTBMRT_properties[] = {
//   // {"device", (getter)THPTStream_get_device, nullptr, nullptr, nullptr},
//   // {"tpu_stream", (getter)THPTStream_get_tpu_stream, nullptr, nullptr, nullptr},
//   // {"priority", (getter)THPTStream_get_priority, nullptr, nullptr, nullptr},
//   {nullptr},
// };

static PyMethodDef THPTBMRT_methods[] = {
  // net
  {"forward",          (PyCFunction)THPTBMRT_forward,             METH_VARARGS, nullptr},
  {"SetRuningNet" ,    (PyCFunction)THPTBMRT_setRuningNet,        METH_VARARGS, nullptr},
  {"GenInplaceTensor", (PyCFunction)THPTBMRT_InPlaceTensor,       METH_VARARGS, nullptr},
  {"GetIONames",       (PyCFunction)THPTBMRT_GetIONames,          METH_VARARGS, nullptr},
  {nullptr}
};

PyTypeObject THPTBMRTType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_tpu._C._TPUBModelRtBase",       /* tp_name */
  sizeof(THPTBMRT),                      /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTBMRT_dealloc,         /* tp_dealloc */
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
  THPTBMRT_methods,                      /* tp_methods */
  0,                      /* tp_members */
  0,                   /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPTBMRT_pynew,                        /* tp_new */
};

PyObject *THPTBMRTClass = nullptr;
void THPTBMRT_init(PyObject *module)
{
#if defined BACKEND_SG2260
  THPTBMRTClass = (PyObject*)&THPTBMRTType;
  if (PyType_Ready(&THPTBMRTType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPTBMRTType);
  if (PyModule_AddObject(module, "_TPUBModelRtBase", (PyObject *)&THPTBMRTType) < 0) {
    throw python_error();
  }
#endif
}