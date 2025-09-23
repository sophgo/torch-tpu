#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"
#include "ops.hpp"

#include <cstdlib>
#include <string>

namespace at {

void set_env(const std::string &env_name, const std::string &env_value) {
    TIMING_START;
    
    // 参数验证
    TORCH_CHECK(!env_name.empty(), "Environment variable name cannot be empty");
    
    // 设置环境变量
    int result = setenv(env_name.c_str(), env_value.c_str(), 1);
    TORCH_CHECK(result == 0, "Failed to set environment variable: ", env_name);
    
    TIMING_END;
}

void clear_env(const std::string &env_name) {
    TIMING_START;
    
    // 参数验证
    TORCH_CHECK(!env_name.empty(), "Environment variable name cannot be empty");
    
    // 清除环境变量
    int result = unsetenv(env_name.c_str());
    TORCH_CHECK(result == 0, "Failed to clear environment variable: ", env_name);
    
    TIMING_END;
}

} // namespace at
