#pragma once

#include <torch/csrc/python_headers.h>
namespace torch_tpu {
namespace autograd {
    void initTorchFunctions(PyObject* module);
}
}