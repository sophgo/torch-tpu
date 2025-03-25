#include <torch/torch.h>
#include <iostream>
#include <vector>

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
    // 设置随机种子以便结果可复现
    torch::manual_seed(42);

    // 创建网络实例
    auto net = std::make_shared<SimpleNet>();

    // 创建优化器（使用 SGD 优化器）
    torch::optim::SGD optimizer(net->parameters(), /*learning_rate=*/0.01);

    // 生成随机数据（输入特征和目标值）
    auto inputs = torch::randn({100, 10}); // 100个样本，每个样本10个特征
    auto targets = torch::randn({100, 1}); // 100个目标值

    // 训练循环
    const int num_epochs = 100; // 训练100个epoch
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 前向传播
        auto outputs = net->forward(inputs);

        // 计算损失（均方误差）
        auto loss = torch::mse_loss(outputs, targets);

        // 打印损失
        if (epoch % 10 == 0) {
            std::cout << "Epoch [" << epoch << "/" << num_epochs
                      << "], Loss: " << loss.item<float>() << std::endl;
        }

        // 反向传播
        optimizer.zero_grad(); // 清空梯度
        loss.backward();       // 计算梯度
        optimizer.step();      // 更新参数
    }

    std::cout << "Training finished!" << std::endl;

    return 0;
}
