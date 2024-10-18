#pragma once
#include <c10/core/TensorOptions.h>

#include "torch_tpu/csrc/core/TPUDeviceManager.h"

namespace torch_tpu {
namespace utils {

void tpu_lazy_init();

void tpu_set_run_yet_variable_to_false();

}; // namespace utils
}; // namespace torch_tpu