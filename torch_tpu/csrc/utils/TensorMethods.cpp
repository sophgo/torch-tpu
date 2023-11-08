#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/tensor_types.h>
#include <c10/core/Backend.h>
#include <torch/csrc/autograd/generated/VariableType.h>

#include "torch_tpu/csrc/utils/TensorMethods.h"
#include "torch_tpu/csrc/utils/DeviceParser.h"
#include "torch_tpu/csrc/utils/LazyInit.h"

namespace torch_tpu{
namespace utils{

const char* _backend_to_string_tpu(const at::Backend& backend) {
    switch (backend) {
        case at::Backend::CPU: return "torch";
        case at_tpu::key::NativeBackend: return "torch.tpu";
        default: AT_ERROR("Unimplemented backend ", backend);
    }
}

std::string _options_to_string_tpu(const at::TensorOptions options) {
  std::ostringstream ss;
  ss << _backend_to_string_tpu(options.backend()) << "." << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

std::string _type_to_string_tpu(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  ss << _backend_to_string_tpu(type.backend()) << "." << toString(type.scalarType()) << "Tensor";
  return ss.str();
}

std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  std::vector<at::DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (int64_t s = 0; s < static_cast<int64_t>(at::ScalarType::NumOptions); s++) {
      auto& type = at::getDeprecatedTypeProperties(static_cast<at::Backend>(p), static_cast<at::ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}

std::vector<at::DeprecatedTypeProperties*> allTPUTypes() {
  return allTypesForBackends({ at_tpu::key::NativeBackend });
}

at::TensorOptions _options_from_string(const std::string& str) {
  static std::string cuda_prefix("torch.cuda.");
  static std::string tpu_prefix("torch.tpu.");
  static std::once_flag cpu_once;
  static std::once_flag cuda_once;
  static std::once_flag tpu_once;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cuda_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> tpu_map;

  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>* map = nullptr;

  if (str == "torch.Tensor") {
    auto backend = c10::dispatchKeyToBackend(torch::tensors::get_default_dispatch_key());
    auto scalar_type = torch::tensors::get_default_scalar_type();
    return at::getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin()).first == cuda_prefix.end()) {
    // torch.cuda. is prefix of str
    std::call_once(cuda_once, []() {
      for (auto type : torch::autograd::VariableType::allCUDATypes()) {
        cuda_map.emplace(torch::utils::type_to_string(*type), type);
      }
    });
    map = &cuda_map;
  } else if (std::mismatch(tpu_prefix.begin(), tpu_prefix.end(), str.begin())
          .first == tpu_prefix.end()) {
    // torch.npu. is prefix of str
    std::call_once(tpu_once, []() {
      for (auto type : allTPUTypes()) {
        tpu_map.emplace(_type_to_string_tpu(*type), type);
      }
    });
    map = &tpu_map;
  } else {
    std::call_once(cpu_once, []() {
      for (auto type : torch::autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(torch::utils::type_to_string(*type), type);
      }
    });
    map = &cpu_map;
  }

  auto it = map->find(str);
  if (it == map->end()) {
    throw torch::ValueError("invalid type: '%s'", str.c_str());
  }
  return it->second->options();
}

std::tuple<at::Tensor, c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool, c10::optional<at::MemoryFormat>>
parse_to_conversion(torch::PythonArgs& r, bool allow_copy);

static at::Tensor dispatch_to(const at::Tensor & self, c10::Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(const at::Tensor & self, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AutoNoGIL no_gil;
  return self.to(self.options().memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(const at::Tensor & self, c10::ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static at::Tensor dispatch_to(const at::Tensor & self, c10::Device device, c10::ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static PyObject* THPVariable_tpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "tpu(Tensor temp, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "tpu(Tensor temp, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto local_device = r.isNone(1) ? c10::Device(at_tpu::key::NativeDeviceType) : r.device(1);
  auto device = c10::Device(at_tpu::key::NativeDeviceType, local_device.index());
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK((device.type() == at_tpu::key::NativeDeviceType), "Invalid device, must be tpu device");
  maybe_initialize_tpu(device);
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "to(Tensor temp, Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor temp, ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor temp, Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  torch::ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto parsed = torch_tpu::utils::parse_to_conversion(r, true);
  auto self_ = std::get<0>(parsed);
  auto& device = std::get<1>(parsed);
  auto& scalarType = std::get<2>(parsed);
  auto non_blocking = std::get<3>(parsed);
  auto copy = std::get<4>(parsed);
  auto opt_memory_format = std::get<5>(parsed);

  maybe_initialize_tpu(device);
  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    Py_INCREF(self);
    return THPVariable_Wrap(self_);
  } else if (!device && !scalarType) {
    return THPVariable_Wrap(
        dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp, PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(Tensor temp, PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.isNone(1)) {
    return THPUtils_packString(_options_to_string_tpu(self_.options()));
  }
  auto obj = r.pyobject(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw torch::TypeError("dtype must be a type, str, or dtype object");
  }
  c10::ScalarType scalar_type;
  c10::Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(1);
  } else {
    at::TensorOptions options = _options_from_string(type_name);
    scalar_type = at::typeMetaToScalarType(options.dtype());
    auto device_type = options.device().type();
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  }
  maybe_initialize_tpu(device);
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_tpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  return torch::autograd::utils::wrap(at_tpu::key::isDeviceTensor(self_));
  END_HANDLE_TH_ERRORS
}

static PyMethodDef TorchTensorMethods[] = {
    {"tpu", castPyCFunctionWithKeywords(THPVariable_tpu), METH_VARARGS | METH_KEYWORDS, NULL},
    {"to", castPyCFunctionWithKeywords(THPVariable_to), METH_VARARGS | METH_KEYWORDS, NULL},
    {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_tpu", castPyCFunctionWithKeywords(THPVariable_is_tpu), METH_VARARGS | METH_KEYWORDS, NULL},
    {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* tensor_functions() {
    return TorchTensorMethods;
}

std::tuple<at::Tensor, c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool, c10::optional<at::MemoryFormat>>
parse_to_conversion(torch::PythonArgs& r, bool allow_copy) {
  if (r.idx == 0) {
    if (!allow_copy && !r.isNone(4))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.tensor(0), at_tpu::key::parse_tpu_device_optional(r.args[1]), r.scalartypeOptional(2), r.toBool(3), r.toBool(4), r.memoryformatOptional(5));
  } else if (r.idx == 1) {
    if (!allow_copy && !r.isNone(4))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.tensor(0), c10::nullopt, r.scalartype(1), r.toBool(2), r.toBool(3), r.memoryformatOptional(4));
  } else {
    auto tensor = r.tensor(1);
    if (!allow_copy && !r.isNone(5))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
        std::move(r.tensor(0)),
        tensor.device(),
        tensor.scalar_type(),
        r.toBool(2),
        r.toBool(3),
        r.memoryformatOptional(4)
    );
  }
}

}; // namespace utils
}; // namespace torch_tpu