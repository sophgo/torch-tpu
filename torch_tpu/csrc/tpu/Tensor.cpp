// #include <torch/csrc/python_headers.h>
// #include <torch/csrc/utils/python_arg_parser.h>
// #include <torch/csrc/tensor/python_tensor.h>
// #include "Tensor.h"

// static PyObject* THPTTensor_FormatCastWrap( PyObject* self, PyObject* tensor_format) {
//     HANDLE_TH_ERRORS
//     THPUtils_assert(
//         THPUtils_checkLong(tensor_format), "invalid format to Format Cast");
//     int64_t format = THPUtils_unpackLong(tensor_format);
//     Py_RETURN_NONE;
//     END_HANDLE_TH_ERRORS
// }

// static PyObject* THPTModule_GetFormatWrap(PyObject* self, PyObject* noargs) {
//     HANDLE_TH_ERRORS
//     // TODO: int64_t format_id = GETFORAMT
//     return PyLong_FromLong(format);
//     END_HANDLE_TH_ERRORS
// }

// static struct PyMethodDef THPTTensor_methods[] = {
//     {"_tpu_format_cast", (PyCFunction)THPTTensor_FormatCastWrap, METH_0, nullptr},
//     {"_tpu_get_format",  (PyCFunction)THPTModule_GetFormatWrap, METH_NOARGS, nullptr},
//     {nullptr}
// };

// PyMethodDef* THPTTensor_get_methods() {
//   return THPTTensor_methods;
// }