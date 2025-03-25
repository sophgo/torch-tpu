#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "aten/TPUGeneratorImpl.h"
#include "TPUDeviceManager.h"
#include "TPUGuard.h"
#include "help.h"



struct ResNet18Impl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    ResidualBlock block1{nullptr}, block2{nullptr}, block3{nullptr}, block4{nullptr}, block5{nullptr}, block6{nullptr}, block7{nullptr}, block8{nullptr};

    torch::nn::Linear fc{nullptr};

    ResNet18Impl() {
        // conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)));
        conv1 = register_module("conv1", conv2d(3, 64, 7, 2, 3));
        pool  = register_module("pool", maxpool2d(3,2,1) );

        // 第一层残差块
        block1 = register_module("block1", ResidualBlock(64, 64));
        block2 = register_module("block2", ResidualBlock(64, 64));

        // 第二层残差块 (下采样)
        block3 = register_module("block3", ResidualBlock(64, 128, true));
        block4 = register_module("block4", ResidualBlock(128, 128));

        // 第三层残差块 (下采样)
        block5 = register_module("block5", ResidualBlock(128, 256, true));
        block6 = register_module("block6", ResidualBlock(256, 256));

        // 第四层残差块 (下采样)
        block7 = register_module("block7", ResidualBlock(256, 512, true));
        block8 = register_module("block8", ResidualBlock(512, 512));

        fc = register_module("fc", torch::nn::Linear(512, 1000));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1(x));
        x = pool(x);
        x = block1->forward(x);
        x = block2->forward(x);
        x = block3->forward(x);
        x = block4->forward(x);
        x = block5->forward(x);
        x = block6->forward(x);
        x = block7->forward(x);
        x = block8->forward(x);
        x = x.mean({2, 3});
        x = fc(x);
        return x;
    }
};


struct ResNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu;
    torch::nn::MaxPool2d maxpool;

    // Define the layers for ResNet-50
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::Linear fc;

    ResNetImpl()
        : conv1(conv2d(3, 64, 7, 2, 3)),
          bn1(torch::nn::BatchNorm2d(64)),
          relu(),
          maxpool(maxpool2d(3, 2, 1)),
          layer1(make_layer(64, 64, 3)),
          layer2(make_layer(64, 128, 4, 2)),
          layer3(make_layer(128, 256, 6, 2)),
          layer4(make_layer(256, 512, 3, 2)),
          fc(torch::nn::Linear(512, 1000)) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("maxpool", maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = maxpool(relu(bn1(conv1(x))));
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = x.mean({2, 3});
        x = fc(x);

        return x;
    }

    torch::nn::Sequential make_layer(int64_t in_channels, int64_t out_channels, int64_t num_blocks, int64_t stride = 1) {
        torch::nn::Sequential layers;
        layers->push_back(ResidualBlock(in_channels, out_channels, stride != 1));

        for (int i = 1; i < num_blocks; ++i) {
            layers->push_back(ResidualBlock(out_channels, out_channels));
        }

        return layers;
    }
};

int main() {
    torch::manual_seed(42);

    // auto net = std::make_shared<ResNet18Impl>();
    auto net = std::make_shared<ResNetImpl>();

    net->to(F16);
    net->to(at::DeviceType::TPU, tpu::TPUGetDeviceIndex());

    torch::optim::SGD optimizer(net->parameters(), 0.01);

    int batch  = 1;

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
