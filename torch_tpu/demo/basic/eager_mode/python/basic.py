import torch
import torch.nn as nn
import torch.optim as optim
import torch_tpu
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity  # 残差连接
        x = self.relu(x)
        return x

# 定义简单卷积网络
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1  = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 16)
        self.fc     = nn.Linear(16 * 14 * 14, 1)  # 假设输入为28x28的图像

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

def main():
    torch.manual_seed(42)

    net = SimpleConvNet().half().to("tpu")
    # print(net)  # 打印网络结构

    # 将模型移动到TPU，如果使用TPU的话
    # net.to('xla:0')  # Uncomment if using TPU with PyTorch/XLA

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    inputs = torch.randn(40, 1, 28, 28).half().to("tpu")  # 输入为100个28x28的单通道图像
    targets = torch.randn(40, 1).to("tpu")

    num_epochs = 100
    for epoch in range(num_epochs):

        outputs = net(inputs)  # 使用 CPU 或 GPU 进行测试
        loss = nn.MSELoss()(outputs, targets)  # 计算损失

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training finished!")

if __name__ == "__main__":
    main()
