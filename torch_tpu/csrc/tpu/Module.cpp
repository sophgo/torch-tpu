#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUException.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"
#include "torch_tpu/csrc/aten/TPUGeneratorImpl.h"
#include "torch_tpu/csrc/utils/LazyInit.h"
#include "torch_tpu/csrc/core/TPUFunction.h"
#if defined BACKEND_SG2260
#include "torch_tpu/csrc/core/TPUStream.h"
#include "ProcessGroupSCCL.hpp"
#include "torch_tpu/csrc/core/Interface/sgrtInterface.h"
#endif

#define CHANGE_UNIT_SIZE 1024.0

struct TPUDeviceProp {
  std::string name;
  size_t totalGlobalMem = 0;
};

TPUDeviceProp prop;
void RegisterTPUProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<TPUDeviceProp>(m, "_TPUDeviceProperties")
            .def_readonly("name", &TPUDeviceProp::name)
            .def_readonly("total_memory", &TPUDeviceProp::totalGlobalMem)
            .def("__repr__", [](const TPUDeviceProp &prop) {
              std::ostringstream stream;
              stream << "_TPUDeviceProperties(name='" << prop.name << "', total_memory="
                << prop.totalGlobalMem / (CHANGE_UNIT_SIZE * CHANGE_UNIT_SIZE) << "MB)";
              return stream.str();
            });
#if defined BACKEND_SG2260
  m.def("createProcessGroupSCCL", &c10d::ProcessGroupSCCL::createProcessGroupSCCL);
  py::class_<c10d::ProcessGroupSCCL::Options>(m, "ProcessGroupSCCLOptions")
    .def(py::init<>())
    .def_readwrite("chip_map", &c10d::ProcessGroupSCCL::Options::chip_map);
#endif
}

TPUDeviceProp* GetDeviceProperties(int64_t deviceid) {
  const char* device_name;
  size_t device_total = 0;
#ifdef BACKEND_1684X
  device_name = "BM1684X";
#elif defined BACKEND_SG2260
  device_name = "SG2260";
#endif
  if (device_name == nullptr) {
    prop.name = " ";
    SOPHON_LOGE("TPU get device name fail.");
  } else {
    prop.name = std::string(device_name);
  }
  //C10_TPU_CHECK(c10_tpu::sgrt::SgrtGetMemInfo(c10_tpu::sgrt::SGRT_GLOBAL_MEM, &device_free, &device_total));
  prop.totalGlobalMem = device_total;
  return &prop;
}

void BindGetDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_tpu_getDeviceProperties", [](int deviceid) -> TPUDeviceProp* {
    return GetDeviceProperties(deviceid);
  }, py::return_value_policy::reference);
}

static PyObject* THPTModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    auto status = tpu::InitTPUMgr();
    if (status != tpu::INIT_SUCCESS) {SOPHON_LOGE("init device failed");}
  }
  auto m = THPObjectPtr(PyImport_ImportModule("torch_tpu.tpu"));
  if (!m) {
      throw python_error();
  }

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };
  auto num_tpus = tpu::TPUGetDeviceCount();
  auto default_tpu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_tpus));
  for (int i = 0; i < num_tpus; i++) {
    auto gen = at_tpu::detail::getDefaultTPUGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_tpu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_tpu_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_set_run_yet_variable_to_false_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_tpu::utils::tpu_set_run_yet_variable_to_false();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

void THPTModule_setDevice(c10::DeviceIndex device) {
  c10_tpu::set_device(device);
}

PyObject* THPTModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackLong(arg);
  {
    pybind11::gil_scoped_release no_gil;
    // init device
    auto status = tpu::InitTPUMgr();
    if (status != tpu::INIT_SUCCESS){  SOPHON_LOGE( "INIT Device Failed"); }
  }
  tpu::TPUSetDeviceIndex(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_tpu::utils::tpu_lazy_init();
  int device = tpu::TPUGetDeviceIndex();
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(tpu::TPUGetDeviceCount());
  END_HANDLE_TH_ERRORS
}

#if defined BACKEND_SG2260
PyObject* THPTModule_tpuSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_release no_gil;
  c10_tpu::tpuSynchronizeDevice();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_tpuFlush(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_release no_gil;

  auto stream = c10_tpu::getCurrentTPUStream();
  tpudnnFlush(stream);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_tpuGetTopology(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_release no_gil;
  PyObject* input_list = nullptr;

  if (!PyArg_ParseTuple(args, "O", &input_list)) {
    return nullptr;
  }

  if (!PyList_Check(input_list)) {
    PyErr_SetString(PyExc_TypeError, "Input must be a list.");
    return nullptr;
  }
  int dev_cnt = tpu::TPUGetDeviceCount();

  if (PyList_Size(input_list) != dev_cnt) {
    PyErr_SetString(PyExc_ValueError, "input_list size does not match device count.");
    return nullptr;
  }

  std::vector<std::vector<int>> topology(dev_cnt, std::vector<int>(dev_cnt));
  tpu::TPUGetTopology(&topology);

  // Assign values from 'topology' to 'input_list'
  pybind11::gil_scoped_acquire acquire_gil;
  for (int i = 0; i < dev_cnt; ++i) {
    PyObject* row = PyList_GetItem(input_list, i);
    if (!PyList_Check(row) || PyList_Size(row) != dev_cnt) {
      PyErr_SetString(PyExc_ValueError, "Each element of input_list must be a list of the correct size.");
      return nullptr;
    }
    for (int j = 0; j < dev_cnt; ++j) {
      PyObject* value = PyLong_FromLong(topology[i][j]); 
      // Replace the existing item in the row with the new value
      if (PyList_SetItem(row, j, value) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set item in input_list.");
        Py_DECREF(value);
        return nullptr;
      }
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_tpuCanDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *value_1 = nullptr;
  PyObject *value_2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &value_1, &value_2)) {
    throw torch::TypeError("Pybind failed to parse parameters.");
  }
  int32_t device_id = THPUtils_unpackInt(value_1);
  int32_t peer_device_id = THPUtils_unpackInt(value_2);
  auto can_access_peer = c10_tpu::sgrt::can_device_access_peer(device_id, peer_device_id);
  return PyBool_FromLong(can_access_peer);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getDeviceUtilizationRate_wrap(PyObject* self, PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(device_index), "invalid argument to getDeviceUtilizationRate");
  //TODO: complete
  int64_t util_rate = 0;
  THPUtils_assert(util_rate <=100 && util_rate >= 0, "invalid result to util_rate");
  return PyLong_FromLong(util_rate);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getCurrentStream_wrap(PyObject* self, PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = c10_tpu::getCurrentTPUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_getDefaultStream_wrap(PyObject *self /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = c10_tpu::getDefaultTPUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
      args,
      kwargs,
      "|LLL",
      const_cast<char**>(kwlist),
      &stream_id,
      &device_index,
      &device_type)) {
  }

  auto stream = c10_tpu::TPUStream::unpack3(
      stream_id, device_index, static_cast<c10::DeviceType>(device_type));

  c10::DeviceIndex device = c10_tpu::current_device();
  if (device != stream.device_index()) {
    THPTModule_setDevice(stream.device_index());
  }
  c10_tpu::setCurrentTPUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::EmptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

#endif

PyObject* THPTModule_tensor_construct_from_storage(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"set_storage_with_format_(Storage source)", },
      /* traceable= */ false
      );

  torch::ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, nullptr, parsed_args);

  at::ScalarType storage_scalar_type;
  bool is_typed_storage = true;
  c10::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
  auto dst_options = c10::TensorOptions().device(storage.device()).dtype(at::kByte);
  auto dst_tensor = at::empty({0}, {}, dst_options).set_(storage);
  return THPVariable_Wrap(dst_tensor);

  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THPTModule_methods[] = {
    {"_tpu_init", (PyCFunction)THPTModule_initExtension, METH_NOARGS, nullptr},
    {"_tpu_set_run_yet_variable_to_false", (PyCFunction)THPTModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
    {"_tpu_setDevice", (PyCFunction)THPTModule_setDevice_wrap, METH_O, nullptr},
    {"_tpu_getDevice", (PyCFunction)THPTModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_tpu_getDeviceCount", (PyCFunction)THPTModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
#if defined BACKEND_SG2260
    {"_tpu_synchronize", (PyCFunction)THPTModule_tpuSynchronize, METH_NOARGS, nullptr},
    {"_tpu_flush", (PyCFunction)THPTModule_tpuFlush, METH_NOARGS, nullptr},
    {"_tpu_getTopology", (PyCFunction)THPTModule_tpuGetTopology, METH_VARARGS, nullptr},
    {"_tpu_canDeviceAccessPeer", (PyCFunction)THPTModule_tpuCanDeviceAccessPeer_wrap, METH_VARARGS, nullptr},
    {"_tpu_getDeviceUtilizationRate", (PyCFunction)THPTModule_getDeviceUtilizationRate_wrap, METH_O, nullptr},
    {"_tpu_getCurrentStream", (PyCFunction)THPTModule_getCurrentStream_wrap, METH_O, nullptr},
    {"_tpu_getDefaultStream", (PyCFunction)THPTModule_getDefaultStream_wrap, METH_O, nullptr},
    {"_tpu_setStream", (PyCFunction)THPTModule_setStream_wrap,  METH_VARARGS | METH_KEYWORDS, nullptr},
    // allocator related
    {"_tpu_emptyCache", THPTModule_emptyCache, METH_NOARGS, nullptr},

#endif
    {"_tensor_construct_from_storage", (PyCFunction)THPTModule_tensor_construct_from_storage, METH_VARARGS, nullptr},
    {nullptr}
};

PyMethodDef* THPTModule_get_methods() {
  return THPTModule_methods;
}
