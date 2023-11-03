#include <Python.h>
#include <torch/torch.h>
#include <torch/csrc/Generator.h>

#include "torch_tpu/csrc/tpu/Generator.h"

bool THPGenerator_init(PyObject *module);

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
    static struct PyModuleDef torchtpu_module = {
        PyModuleDef_HEAD_INIT,
        "torch_tpu._C",
        nullptr,
        -1,
        methods.data()
    };
    module = PyModule_Create(&torchtpu_module);
    THPGenerator_init(module);
    return module;
}
PyMODINIT_FUNC PyInit__C(void){
    return initModule();
}


}; //extern "C"