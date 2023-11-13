#include <Python.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/out_types.h>

#include "torch_tpu/csrc/utils/DeviceParser.h"
#include "torch_tpu/csrc/utils/LazyInit.h"
#include "torch_tpu/csrc/aten/python_custom_functions.h"

namespace torch_tpu { 
namespace autograd {

static PyObject * THPVariableFunctionsModule = NULL;

// ----- declarations -----
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject* THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs);
// ----- declarations -----

static PyMethodDef torch_functions[] = {
    {"randn", castPyCFunctionWithKeywords(THPVariable_randn), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {"randn_like", castPyCFunctionWithKeywords(THPVariable_randn_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {"rand", castPyCFunctionWithKeywords(THPVariable_rand), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {"rand_like", castPyCFunctionWithKeywords(THPVariable_rand_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {"randint", castPyCFunctionWithKeywords(THPVariable_randint), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {"randint_like", castPyCFunctionWithKeywords(THPVariable_randint_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {"randperm", castPyCFunctionWithKeywords(THPVariable_randperm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {NULL}
};

static PyTypeObject THPVariableFunctions = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "torch_tpu._C._VariableFunctionsClass", /*tp_name*/
    0,                                     /* tp_basicsize */
    0,                                     /* tp_itemsize */
    0,                                     /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT,                    /* tp_flags */
    NULL,                                  /* tp_doc */
    0,                                     /* tp_traverse */
    0,                                     /* tp_clear */
    0,                                     /* tp_richcompare */
    0,                                     /* tp_weaklistoffset */
    0,                                     /* tp_iter */
    0,                                     /* tp_iternext */
    torch_functions,                       /* tp_methods */
    0,                                     /* tp_members */
    0,                                     /* tp_getset */
    0,                                     /* tp_base */
    0,                                     /* tp_dict */
    0,                                     /* tp_descr_get */
    0,                                     /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    0,                                     /* tp_init */
    0,                                     /* tp_alloc */
    0                                      /* tp_new */
};

void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0){
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  if (PyModule_AddObject(module, "_VariableFunctionsClass", reinterpret_cast<PyObject*>(&THPVariableFunctions))){
    throw python_error();
  }

  THPVariableFunctionsModule = PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  if (PyModule_AddObject(module, "_VariableFunctions", THPVariableFunctionsModule) < 0){
    throw python_error();
  }
}
// ---- define ----
using namespace torch::autograd::utils;
using namespace torch::utils;
using at::DimnameList;
using at::TensorOptions;
using at::Tensor;
using at::IntArrayRef;
using at::Generator;


inline Tensor dispatch_randint(int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size, generator);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  torch_tpu::utils::maybe_initialize_tpu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch_tpu::utils::maybe_initialize_tpu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size, generator);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  torch_tpu::utils::maybe_initialize_tpu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch_tpu::utils::maybe_initialize_tpu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, options);
}

// randn
static PyObject* THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    static torch::PythonArgParser parser ({
      "randn(IntArrayRef size, *, Generator? generator, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
      "randn(IntArrayRef size, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
      "randn(IntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
      "randn(IntArrayRef size, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    }, /*traceable=*/true);
    torch::ParsedArgs<8> parsed_args;
    auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
    if (_r.has_torch_function()){
        return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
    }
    switch (_r.idx) {
        case 0 : {
          // aten::randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
          auto __names = _r.toDimnameListOptional(2);
          c10::optional<at::DimnameList> names = __names ? c10::make_optional(at::DimnameList(__names.value())) : c10::nullopt;
          const auto options = at::TensorOptions()
                                  .dtype(_r.scalartypeOptional(3))
                                  .device(at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()))
                                  .layout(_r.layoutOptional(4))
                                  .requires_grad(_r.toBool(7))
                                  .pinned_memory(_r.toBool(6));
          // init TODO

          auto dispatch_randn = [](at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
            pybind11::gil_scoped_release no_gil;
            // RECORD_FUNCTION("randn")
            return torch::randn(size, generator, names, options);
          };
          return wrap(dispatch_randn(_r.intlist(0), _r.generator(1), names, options));
        }         
        case 1 : {
          if (_r.isNone(2)) {
            // aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
            const auto options = at::TensorOptions()
                .dtype(_r.scalartypeOptional(3))
                .device(at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()))
                .layout(_r.layoutOptional(4))
                .requires_grad(_r.toBool(7))
                .pinned_memory(_r.toBool(6));
            auto dispatch_randn = [](at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
              pybind11::gil_scoped_release no_gil;
              // RECORD_FUNCTION("randn")
              return torch::randn(size, generator, options);
            };
            return wrap(dispatch_randn(_r.intlist(0), _r.generator(1), options));
          } else {
            // aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
            check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()), _r.isNone(5));
            auto dispatch_randn_out = [](at::Tensor out, at::IntArrayRef size, c10::optional<at::Generator> generator) -> at::Tensor {
              pybind11::gil_scoped_release no_gil;
              // RECORD_FUNCTION("randn_out")              
              return at::randn_out(out, size, generator);
            };
            return wrap(dispatch_randn_out(_r.tensor(2), _r.intlist(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
          }
        }
        case 2: {
          if (_r.isNone(1)) {
            // aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
            const auto options = at::TensorOptions()
                .dtype(_r.scalartypeOptional(2))
                .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()))
                .layout(_r.layoutOptional(3))
                .requires_grad(_r.toBool(6))
                .pinned_memory(_r.toBool(5));
            //torch_tpu::utils::maybe_initialize_tpu(options);
            
            auto dispatch_randn = [](at::IntArrayRef size, at::TensorOptions options) -> at::Tensor {
              pybind11::gil_scoped_release no_gil;              
              return torch::randn(size, options);
            };
            return wrap(dispatch_randn(_r.intlist(0), options));
          } else {
            // aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
            check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                                  _r.isNone(2), _r.layoutOptional(3),
                                  at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()), _r.isNone(4));
            
            auto dispatch_randn_out = [](at::Tensor out, at::IntArrayRef size) -> at::Tensor {
              //
              pybind11::gil_scoped_release no_gil;
              // RECORD_FUNCTION("randn_out")
              return at::randn_out(out, size);
            };
            return wrap(dispatch_randn_out(_r.tensor(1), _r.intlist(0)).set_requires_grad(_r.toBool(6)));
          }
        }
        case 3: {
          // aten::randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
          auto __names = _r.toDimnameListOptional(1);
          c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
          const auto options = at::TensorOptions()
              .dtype(_r.scalartypeOptional(2))
              .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()))
              .layout(_r.layoutOptional(3))
              .requires_grad(_r.toBool(6))
              .pinned_memory(_r.toBool(5));
          //torch_tpu::utils::maybe_initialize_tpu(options);
          
          auto dispatch_randn = [](at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
            pybind11::gil_scoped_release no_gil;
            // RECORD_FUNCTION("randn")
            return torch::randn(size, names, options);
          };
          return wrap(dispatch_randn(_r.intlist(0), names, options));
        }
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

// randn_like
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "randn_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);
  torch::ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
      .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], self.device()))
      .layout(_r.layoutWithDefault(3, self.layout()))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch_tpu::utils::maybe_initialize_tpu(options);
  
  auto dispatch_randn_like = [](const at::Tensor & self, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    //
    pybind11::gil_scoped_release no_gil;
    // RECORD_FUNCTION("randn_like")
    
    return torch::randn_like(self, options, memory_format);
  };
  return wrap(dispatch_randn_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// randperm
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "randperm(int64_t n, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randperm(int64_t n, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);
  torch::ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch_tpu::utils::maybe_initialize_tpu(options);
        
        auto dispatch_randperm = [](int64_t n, c10::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("randperm")
          
          return torch::randperm(n, generator, options);
        };
        return wrap(dispatch_randperm(_r.toInt64(0), _r.generator(1), options));
      } else {
        // aten::randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_randperm_out = [](at::Tensor out, int64_t n, c10::optional<at::Generator> generator) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("randperm_out")
          
          return at::randperm_out(out, n, generator);
        };
        return wrap(dispatch_randperm_out(_r.tensor(2), _r.toInt64(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch_tpu::utils::maybe_initialize_tpu(options);
        
        auto dispatch_randperm = [](int64_t n, at::TensorOptions options) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("randperm")
          
          return torch::randperm(n, options);
        };
        return wrap(dispatch_randperm(_r.toInt64(0), options));
      } else {
        // aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_randperm_out = [](at::Tensor out, int64_t n) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("randperm_out")
          
          return at::randperm_out(out, n);
        };
        return wrap(dispatch_randperm_out(_r.tensor(1), _r.toInt64(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "randint(int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
      }, false);

  torch::ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    auto device = at_tpu::key::parse_tpu_device(r.args[6]);
    if (r.isNone(3)) {
      auto high = r.toInt64(0);
      auto size = r.intlist(1);
      auto generator = r.generator(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(4, at::ScalarType::Long);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5))
          .requires_grad(r.toBool(7));
      return torch::autograd::utils::wrap(dispatch_randint(high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), device, r.isNone(6));
      return torch::autograd::utils::wrap(dispatch_randint(r.toInt64(0), r.intlist(1), r.generator(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  } else if (r.idx == 1) {
    auto device = at_tpu::key::parse_tpu_device(r.args[7]);
    if (r.isNone(4)) {
      auto low = r.toInt64(0);
      auto high = r.toInt64(1);
      auto size = r.intlist(2);
      auto generator = r.generator(3);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(5, at::ScalarType::Long);

      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(6))
          .requires_grad(r.toBool(8));
      return torch::autograd::utils::wrap(dispatch_randint(low, high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(4), r.scalartype(5), r.isNone(5), r.layout(6), device, r.isNone(7));
      return torch::autograd::utils::wrap(dispatch_randint(r.toInt64(0), r.toInt64(1), r.intlist(2), r.generator(3), r.tensor(4)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// randint_like
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "randint_like(Tensor input, int64_t high, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randint_like(Tensor input, int64_t low, int64_t high, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);
  torch::ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(3, self.scalar_type()))
          .device(at_tpu::key::parse_tpu_device_with_default(_r.args[5], self.device()))
          .layout(_r.layoutWithDefault(4, self.layout()))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch_tpu::utils::maybe_initialize_tpu(options);
      
      auto dispatch_randint_like = [](const at::Tensor & self, int64_t high, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) -> at::Tensor {
        //
        pybind11::gil_scoped_release no_gil;
        // RECORD_FUNCTION("randint_like")
        
        return torch::randint_like(self, high, options, memory_format);
      };
      return wrap(dispatch_randint_like(self, _r.toInt64(1), options, _r.memoryformatOptional(2)));
    }
    case 1: {
      // aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeWithDefault(4, self.scalar_type()))
          .device(at_tpu::key::parse_tpu_device_with_default(_r.args[6], self.device()))
          .layout(_r.layoutWithDefault(5, self.layout()))
          .requires_grad(_r.toBool(8))
          .pinned_memory(_r.toBool(7));
      torch_tpu::utils::maybe_initialize_tpu(options);
      
      auto dispatch_randint_like = [](const at::Tensor & self, int64_t low, int64_t high, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) -> at::Tensor {
        //
        pybind11::gil_scoped_release no_gil;
        // RECORD_FUNCTION("randint_like")
        
        return torch::randint_like(self, low, high, options, memory_format);
      };
      return wrap(dispatch_randint_like(self, _r.toInt64(1), _r.toInt64(2), options, _r.memoryformatOptional(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rand_like
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "rand_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);
  torch::ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(2, self.scalar_type()))
      .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], self.device()))
      .layout(_r.layoutWithDefault(3, self.layout()))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch_tpu::utils::maybe_initialize_tpu(options);
  
  auto dispatch_rand_like = [](const at::Tensor & self, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    //
    pybind11::gil_scoped_release no_gil;
    // RECORD_FUNCTION("rand_like")
    
    return torch::rand_like(self, options, memory_format);
  };
  return wrap(dispatch_rand_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rand
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "rand(IntArrayRef size, *, Generator? generator, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "rand(IntArrayRef size, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "rand(IntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "rand(IntArrayRef size, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);
  torch::ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch_tpu::utils::maybe_initialize_tpu(options);
      
      auto dispatch_rand = [](at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        //
        pybind11::gil_scoped_release no_gil;
        // RECORD_FUNCTION("rand")
        
        return torch::rand(size, generator, names, options);
      };
      return wrap(dispatch_rand(_r.intlist(0), _r.generator(1), names, options));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch_tpu::utils::maybe_initialize_tpu(options);
        
        auto dispatch_rand = [](at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("rand")
          
          return torch::rand(size, generator, options);
        };
        return wrap(dispatch_rand(_r.intlist(0), _r.generator(1), options));
      } else {
        // aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               at_tpu::key::parse_tpu_device_with_default(_r.args[5], torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_rand_out = [](at::Tensor out, at::IntArrayRef size, c10::optional<at::Generator> generator) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("rand_out")
          
          return at::rand_out(out, size, generator);
        };
        return wrap(dispatch_rand_out(_r.tensor(2), _r.intlist(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 2: {
      if (_r.isNone(1)) {
        // aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch_tpu::utils::maybe_initialize_tpu(options);
        
        auto dispatch_rand = [](at::IntArrayRef size, at::TensorOptions options) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("rand")
          
          return torch::rand(size, options);
        };
        return wrap(dispatch_rand(_r.intlist(0), options));
      } else {
        // aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_rand_out = [](at::Tensor out, at::IntArrayRef size) -> at::Tensor {
          //
          pybind11::gil_scoped_release no_gil;
          // RECORD_FUNCTION("rand_out")
          
          return at::rand_out(out, size);
        };
        return wrap(dispatch_rand_out(_r.tensor(1), _r.intlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
    case 3: {
      // aten::rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(at_tpu::key::parse_tpu_device_with_default(_r.args[4], torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch_tpu::utils::maybe_initialize_tpu(options);
      
      auto dispatch_rand = [](at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        //
        pybind11::gil_scoped_release no_gil;
        // RECORD_FUNCTION("rand")
        
        return torch::rand(size, names, options);
      };
      return wrap(dispatch_rand(_r.intlist(0), names, options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ---- define ----
}; // namespace autograd
}; // namespace torch_tpu