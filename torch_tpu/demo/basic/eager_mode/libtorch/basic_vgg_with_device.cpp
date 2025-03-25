#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "aten/TPUGeneratorImpl.h"
#include "TPUDeviceManager.h"
#include "TPUGuard.h"
#include "help.h"

struct VGG16Impl : torch::nn::Module {
    torch::nn::Conv2d conv1_1, conv1_2;
    torch::nn::MaxPool2d pool1;

    torch::nn::Conv2d conv2_1, conv2_2;
    torch::nn::MaxPool2d pool2;

    torch::nn::Conv2d conv3_1, conv3_2, conv3_3;
    torch::nn::MaxPool2d pool3;

    torch::nn::Conv2d conv4_1, conv4_2, conv4_3;
    torch::nn::MaxPool2d pool4;

    torch::nn::Conv2d conv5_1, conv5_2, conv5_3;
    torch::nn::MaxPool2d pool5;

    torch::nn::Linear fc1, fc2, fc3;
    torch::nn::Dropout dropout;

    VGG16Impl()
        : conv1_1(register_module("conv1_1", conv2d(3,  64, 3, 1, 1, true))),
          conv1_2(register_module("conv1_2", conv2d(64, 64, 3, 1, 1, true))),
          pool1(register_module("pool1", maxpool2d(2, 2))),

          conv2_1(register_module("conv2_1", conv2d(64, 128,  3, 1, 1, true))),
          conv2_2(register_module("conv2_2", conv2d(128, 128, 3, 1, 1, true))),
          pool2(register_module("pool2", maxpool2d(2, 2))),

          conv3_1(register_module("conv3_1", conv2d(128, 256, 3, 1, 1, true))),
          conv3_2(register_module("conv3_2", conv2d(256, 256, 3, 1, 1, true))),
          conv3_3(register_module("conv3_3", conv2d(256, 256, 3, 1, 1, true))),
          pool3(register_module("pool3", maxpool2d(2, 2))),

          conv4_1(register_module("conv4_1", conv2d(256, 512, 3, 1, 1, true))),
          conv4_2(register_module("conv4_2", conv2d(512, 512, 3, 1, 1, true))),
          conv4_3(register_module("conv4_3", conv2d(512, 512, 3, 1, 1, true))),
          pool4(register_module("pool4", maxpool2d(2, 2))),

          conv5_1(register_module("conv5_1", conv2d(512, 512, 3, 1, 1, true))),
          conv5_2(register_module("conv5_2", conv2d(512, 512, 3, 1, 1, true))),
          conv5_3(register_module("conv5_3", conv2d(512, 512, 3, 1, 1, true))),
          pool5(register_module("pool5", maxpool2d(2, 2))),

          fc1(register_module("fc1", torch::nn::Linear(512 * 7 * 7, 4096))),
          fc2(register_module("fc2", torch::nn::Linear(4096, 4096))),
          fc3(register_module("fc3", torch::nn::Linear(4096, 1000))),
          dropout(register_module("dropout", torch::nn::Dropout(0.5))) {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1_1(x));
        x = torch::relu(conv1_2(x));
        x = pool1(x);

        x = torch::relu(conv2_1(x));
        x = torch::relu(conv2_2(x));
        x = pool2(x);

        x = torch::relu(conv3_1(x));
        x = torch::relu(conv3_2(x));
        x = torch::relu(conv3_3(x));
        x = pool3(x);

        x = torch::relu(conv4_1(x));
        x = torch::relu(conv4_2(x));
        x = torch::relu(conv4_3(x));
        x = pool4(x);

        x = torch::relu(conv5_1(x));
        x = torch::relu(conv5_2(x));
        x = torch::relu(conv5_3(x));
        x = pool5(x);

        x = x.view({x.size(0), -1}); // Flatten
        x = torch::relu(fc1(x));
        x = dropout(x);
        x = torch::relu(fc2(x));
        x = dropout(x);
        x = fc3(x);

        return x; // 返回最终输出
    }
};


int main() {
    torch::manual_seed(42);

    auto net = std::make_shared<VGG16Impl>();

    net->to(F16);
    net->to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());

    torch::optim::SGD optimizer(net->parameters(), 0.01);

    int batch  = 32;

    auto inputs = torch::randn({batch, 3, 224, 224}).to(F16);
    auto targets = torch::randn({batch}).to(Long);

    const int num_epochs = 100;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto inputs_tpu  = inputs.to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());
        auto targets_tpu = targets.to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());

        auto outputs = net->forward(inputs_tpu);

        auto loss = torch::nn::functional::cross_entropy(outputs, targets_tpu);

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