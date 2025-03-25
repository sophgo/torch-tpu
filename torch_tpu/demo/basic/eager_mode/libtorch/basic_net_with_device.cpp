#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "aten/TPUGeneratorImpl.h"
#include "TPUDeviceManager.h"
#include "TPUGuard.h"

struct SimpleNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    SimpleNet() {
        fc1 = register_module("fc1", torch::nn::Linear(10, 32));
        fc2 = register_module("fc2", torch::nn::Linear(32, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }
};

int main() {
    torch::manual_seed(42);

    auto net = std::make_shared<SimpleNet>();

    net->to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());

    torch::optim::SGD optimizer(net->parameters(), 0.01);

    auto inputs = torch::randn({100, 10});
    auto targets = torch::randn({100, 1});

    const int num_epochs = 100;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto inputs_tpu  = inputs.to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());
        auto targets_tpu = targets.to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());

        auto outputs = net->forward(inputs_tpu);

        auto loss = torch::mse_loss(outputs, targets_tpu);

        if (epoch % 10 == 0) {
            std::cout << "Epoch [" << epoch << "/" << num_epochs
                      << "], Loss: " << loss.item<float>() << std::endl;
        }

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    std::cout << "Training finished!" << std::endl;

    return 0;
}
