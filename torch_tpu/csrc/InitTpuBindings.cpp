#include <Python.h>
#include <torch/torch.h>
#include <torch/csrc/Generator.h>

#include "torch_tpu/csrc/tpu/Module.h"
#include "torch_tpu/csrc/tpu/UtilsFunc.h"
#include "torch_tpu/csrc/utils/AutocastMode.h"

#ifdef BACKEND_SG2260
#include "torch_tpu/csrc/tpu/Stream.h"
#include "torch_tpu/csrc/tpu/Event.h"
#include "torch_tpu/csrc/tpu/Bmodel.h"
#endif

PyObject* module;
static std::vector<PyMethodDef> methods;

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

extern "C" {

PyObject* initModule() {
    at::internal::lazy_init_num_threads();

    AddPyMethodDefs(methods, THPTModule_get_methods());
    AddPyMethodDefs(methods, THPTUtils_get_methods());
    AddPyMethodDefs(methods, torch_tpu::autocast::autocast_mode_functions());
#ifdef BACKEND_SG2260
    AddPyMethodDefs(methods, THPTBmodel_get_methods());
#endif
    static struct PyModuleDef torchtpu_module = {
        PyModuleDef_HEAD_INIT,
        "torch_tpu._C",
        nullptr,
        -1,
        methods.data()
    };
    module = PyModule_Create(&torchtpu_module);
#ifdef BACKEND_SG2260
    THPTStream_init(module);
    THPTEvent_init(module);
#endif
    RegisterTPUProperties(module);
    BindGetDeviceProperties(module);
    return module;
}

void deinit() {
#ifdef BACKEND_SG2260
  tpu::TPUDeleteInstance();
#endif
}

PyMODINIT_FUNC PyInit__C(void){
    Py_AtExit(&deinit);
    return initModule();
}
}; //extern "C"
