# Torch-TPU

Torch-TPU 是一个 PyTorch 扩展，使得 PyTorch 模型能够在算能 TPU 设备上运行。

## 目录

- [Torch-TPU](#torch-tpu)
  - [目录](#目录)
  - [特性](#特性)
  - [快速开始](#快速开始)
  - [开发](#开发)
    - [目录结构](#目录结构)
    - [下载所需仓库](#下载所需仓库)
    - [环境准备](#环境准备)
    - [构建和安装](#构建和安装)
    - [使用方法](#使用方法)
    - [高级特性](#高级特性)
      - [JIT 模式 (仅支持 SG2260)](#jit-模式-仅支持-sg2260)
      - [存储格式](#存储格式)
      - [支持PPL](#支持ppl)
  - [其他训练框架示例](#其他训练框架示例)
  - [许可证](#许可证)

## 特性

- 在算能 TPU 设备上执行 PyTorch 模型
- 支持 JIT 和 Eager 执行模式
- 灵活的存储格式选项
- 集成主流深度学习框架：
  - DeepSpeed Zero Stage 1 和 2，支持 CPU 卸载
  - Megatron 张量并行
  - 支持大模型如 Qwen2

## 快速开始

1. 进入 Docker 容器：
```bash
docker run -it  \
        --user `id -u`:`id -g` --privileged --cap-add SYS_ADMIN \
        --name torch_tpu \
        --env HOME=$HOME \
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
        -v $HOME:$HOME \
        -v /workspace:/workspace \
        -v /opt:/opt \
        sophgo/torch_tpu:v0.1-py311  /bin/bash
```

2. 安装 torch-tpu

```bash
pip install torch-tpu
```

3. 准备示例代码：

```python
import torch
import torch_tpu

device = "tpu:0"

a = torch.randn(3, 3).to(device)
b = torch.randn(3, 3).to(device)
c = a + b
print(c)
```

你将看到类似以下输出：

```
2024-12-10 19:57:38,880 - torch_tpu.utils.apply_revert_patch - INFO - Patch directory: /workspace/tpu-train/torch_tpu/utils/../demo/patch
...
Stream created for device 0
tensor([[ 0.5913, -2.9734, -3.1663],
        [-1.0463, -0.4574, -1.6539],
        [-0.5939,  0.9554, -1.2390]])
```

## 开发

### 目录结构

```
/workspace/
├── tpuv7-runtime
├── TPU1686
└── tpu-train
```

### 下载所需仓库

```bash
# 安装 git-lfs
sudo apt install git-lfs

# 使用这些环境变量来避免 git-lfs 错误
# export GIT_SSL_NO_VERIFY=1
# export GIT_LFS_SKIP_SMUDGE=1

# 在 `/workspace` 目录下克隆仓库
git clone <tpu-train-repo-url> /workspace/tpu-train
git clone <tpuv7-runtime-repo-url> /workspace/tpuv7-runtime
git clone <TPU1686-repo-url> /workspace/TPU1686
```

### 环境准备

下载所需仓库：

```bash
git clone <tpu-train-repo-url> /workspace/tpu-train
git clone <tpuv7-runtime-repo-url> /workspace/tpuv7-runtime
git clone <TPU1686-repo-url> /workspace/TPU1686
```

### 构建和安装

对于需要修改源码并构建的开发者：

```bash
cd tpu-train
source scripts/envsetup.sh sg2260

# 构建 TPU 基础内核
rebuild_TPU1686

# 以可编辑模式安装 torch-tpu
develop_torch_tpu
```

RV version:
```bash
cd tpu-train
source scripts/envsetup.sh sg2260

# Debug TPU1686, optional
export EXTRA_CONFIG='-DDEBUG=ON -DUSING_FW_PRINT=ON -DUSING_FW_DEBUG=ON'

# Build TPU base kernels
rebuild_TPU1686_riscv

# Build TPU RISCV base kernels
build_riscv_whl

# 如果失败，尝试更新 bm_prebuilt_toolchains
```

如果一切顺利，现在我们就有了一个可编辑的开发安装。

+ 对于 .py 文件的修改，无需重新安装，可直接生效。
+ 如果修改了 torch-tpu 扩展的 cpp 文件，进入 `build/torch-tpu` 目录执行 `make install` 即可。
+ 如果修改了 `firmware_core` 中的内核源文件，进入 `build/firmware_sg2260[_cmodel]` 目录执行 `make`。

### 使用方法

```shell
python examples/hello_world.py
```

### 高级特性

#### JIT 模式 (仅支持 SG2260)

Torch-TPU 默认使用 JIT 模式。要使用 Eager 模式：
- 找到 `__init__.py` 中的 `TPU_CACHE_BACKEND`
- 将其注释掉

#### 存储格式
要对卷积权重使用 32IC 格式：
```bash
export TORCHTPU_STORAGE_CAST=ON
```

#### 支持PPL

可以使用PPL来开发后端算子，可以选择使用ppl工程或者release包，以SG2260E为例进行说明：

```bash
source scripts/envsetup.sh sg2260e
export PPL_INSTALL_PATH=/the/path/to/ppl/install/
rebuild_TPU1686
develop_torch_tpu
```

## 其他训练框架示例

本仓库包含多个示例实现，展示了不同的分布式训练框架：

| 示例                                    | 位置                                | 源框架              |
| -------------------------------------- | ---------------------------------- | ------------------ |
| [Qwen2-7B](./examples/qwen2/README.md) | [examples/qwen2](./examples/qwen2/README.md) | [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) |

<!-- | [BERT](./examples/bert/README.md)      | [examples/bert](./examples/bert)   |            |
| [GPT](./examples/gpt/README.md)        | [examples/gpt](./examples/gpt)     |            |
| [ViT](./examples/vit/README.md)        | [examples/vit](./examples/vit)     |            | -->

查看 [examples 目录](./examples) 获取每个框架的详细说明和配置。

## 许可证

TPU-TRAIN 采用 2-Clause BSD 许可证，但第三方组件除外。