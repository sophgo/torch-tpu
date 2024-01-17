
## Torch-TPU介绍


## 简介
Torch TPU基于Pytorch的`Privateuseone`后端进行扩展，使算能TPU适配到Pytorch框架，为使用PyTorch框架的开发者提供算能AI处理器的超强算力。

用户可以使用PyTorch的原生接口，在算能的TPU设备上实现加速运行。

## 目录结构与说明
| 目录                                                          | 任务类别描述   | 支持模型          | 可用精度    |
|---                                                           |---           |---               |---      |
| [mnist](./sample/mnist/README.md)                            | 图像分类       | 手造小模型          | FP32/FP16 |
| [StableDiffusion](./sample/StableDiffusion/README.md)        | 文生图         | SD14             | FP32/FP16 |


## 版本说明
| 版本    | 说明 | 
|---     |---   |
| 0.1.0	 | 提供mnist图像识别，StableDiffusion lora finetune 2个例程，适配BM1684X(x86 PCIe) |

## 环境依赖
Torch TPU主要依赖libsophon、python、Pytorch，其版本要求如下:
|操作系统     |libsophon|python      |Pytorch  | 发布日期  |
|---         |-------- |------------| --------|---------|
|ubuntu 22.04| >=0.4.9 | ==3.10     | ==2.1.0 | >=24.01.17|

> **注意**：
> 1. 目前仅支持torch2.1,BM1684X的板卡模态。

## 安装说明
### 方式1 使用whl包安装
请确保满足上述环境依赖。
```
    pip install torch_tpu-2.1.0.post1-cp310-cp310-linux_x86_64.whl
```
### 方式2 在安装好torch-tpu的docker中，直接使用
TODO

安装完之后，可以通过下面的python脚本进行检验，
```python
import torch
import torch_tpu
import torch.nn as nn
device = "tpu"
batch = 8
sequence = 1024
hidden_size = 768
out_size = 3

inp = torch.rand(batch, sequence, hidden_size).to(device)
ln_net = nn.Linear(hidden_size, out_size).to(device)
out = ln_net(inp)
print(out.cpu())
```

## 技术资料

TODO

## 社区

算能社区鼓励开发者多交流，共学习。开发者可以通过以下渠道进行交流和学习。

算能社区网站：https://www.sophgo.com/

算能开发者论坛：https://developer.sophgo.com/forum/index.html


## 贡献

TODO

## 许可证
TODO