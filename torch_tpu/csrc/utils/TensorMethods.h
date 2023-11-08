#include <torch/csrc/python_headers.h>
#include <c10/core/DeviceType.h>

namespace torch_tpu {
namespace utils {

const char* _backend_to_string_npu(const at::Backend& backend);

PyMethodDef* tensor_functions();

}
}