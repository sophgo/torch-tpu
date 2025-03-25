#include <torch/torch.h>
#include <iostream>
#include "aten/TPUGeneratorImpl.h"
#include "TPUDeviceManager.h"
#include "TPUGuard.h"
int main() {
    // 创建两个张量
    torch::Tensor tensor1 = torch::tensor({1.0, 2.0, 3.0});
    torch::Tensor tensor2 = torch::tensor({4.0, 5.0, 6.0});

    auto tpu_tensor1 = tensor1.to( at::DeviceType::TPU, tpu::TPUGetDeviceIndex() );
    auto tpu_tensor2 = tensor2.to( at::DeviceType::TPU, tpu::TPUGetDeviceIndex() );
    auto tensor3 = tensor1 + tensor2;
    // cpy into cpu
    auto cpu_tensor3 = tensor3.to(torch::Device("cpu"));
    std::cout << "tensor 3: " << cpu_tensor3 << std::endl;
    return 0;
}
