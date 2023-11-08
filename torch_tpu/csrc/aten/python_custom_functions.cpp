#include <Python.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/out_types.h>

#include "torch_tpu/csrc/utils/DeviceParser.h"
#include "torch_tpu/csrc/aten/python_custom_functions.h"

namespace torch_tpu { 
namespace autograd {

static PyObject * THPVariableFunctionsModule = NULL;

// ----- declarations -----
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs);
// ----- declarations -----

static PyMethodDef torch_functions[] = {
    {"randn", castPyCFunctionWithKeywords(THPVariable_randn), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
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
                                  .device(at_tpu::key::parse_device_with_default(_r.args[5], torch::tensors::get_default_device()))
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
                .device(at_tpu::key::parse_device_with_default(_r.args[5], torch::tensors::get_default_device()))
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
                               at_tpu::key::parse_device_with_default(_r.args[5], torch::tensors::get_default_device()), _r.isNone(5));
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
                .device(at_tpu::key::parse_device_with_default(_r.args[4], torch::tensors::get_default_device()))
                .layout(_r.layoutOptional(3))
                .requires_grad(_r.toBool(6))
                .pinned_memory(_r.toBool(5));
            //torch_tpu::utils::maybe_initialize_npu(options);
            
            auto dispatch_randn = [](at::IntArrayRef size, at::TensorOptions options) -> at::Tensor {
              pybind11::gil_scoped_release no_gil;              
              return torch::randn(size, options);
            };
            return wrap(dispatch_randn(_r.intlist(0), options));
          } else {
            // aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
            check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                                  _r.isNone(2), _r.layoutOptional(3),
                                  at_tpu::key::parse_device_with_default(_r.args[4], torch::tensors::get_default_device()), _r.isNone(4));
            
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
              .device(at_tpu::key::parse_device_with_default(_r.args[4], torch::tensors::get_default_device()))
              .layout(_r.layoutOptional(3))
              .requires_grad(_r.toBool(6))
              .pinned_memory(_r.toBool(5));
          //torch_npu::utils::maybe_initialize_npu(options);
          
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
// ---- define ----
}; // namespace autograd
}; // namespace torch_tpu