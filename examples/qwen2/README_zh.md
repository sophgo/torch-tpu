[English](./README.md) | [中文](./README_zh.md)

# 使用 PAI-Megatron-Patch 预训练 Qwen2-7B

## 步骤1: 准备 docker 容器

- 使用 sophgo/torch_tpu:latest 镜像创建 docker 容器，并确保安装了最新版本的 torch_tpu 和 tpuv7-runtime 库。
```
docker run
```

容器启动后，可以通过以下命令进入容器：

```
docker exec -it <container_id> /bin/bash
```

## 步骤2: 在 docker 容器中安装 `torch_tpu`

### 可选方式1: 通过 pip 安装 `torch_tpu`

```
pip install torch_tpu
```

### 可选方式2: 直接基于 Release 链接安装 `torch_tpu`

```
<!-- TODO -->
pip install https://github.com/sophgo/torch-tpu/releases/download/v0.1.0/torch_tpu-0.1.0-cp310-cp310-linux_x86_64.whl
```

## 步骤3: 获取 PAI-Megatron-Patch

### 可选方式1: 通过 git apply 获取代码

- 克隆 Pai-Megatron-Patch 仓库

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git /workspace/Pai-Megatron-Patch
cd /workspace/Pai-Megatron-Patch
git checkout v0.9.1
```

- 应用补丁
```bash
cp Pai-Megatron-Sophgo.patch  /workspace/Pai-Megatron-Patch/
cd /workspace/Pai-Megatron-Patch
git apply Pai-Megatron-Sophgo.patch
```

### 可选方式2: 通过发布的演示获取代码

<!-- TODO: 添加发布演示的链接 -->
- 你也可以在[这里](https://github.com/sophgo/torch-tpu/)下载完整的 PAI-Megatron-Patch

## 步骤4: 安装 megatron-patch

获取 PAI-Megatron-Patch 后，需要在 Pai-Megatron-Patch 目录下运行以下命令安装 `megatron-patch`：

```bash
pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets
cd /workspace/Pai-Megatron-Patch/PAI-Megatron-LM-240718
python setup.py develop --user
```

可以通过运行以下命令检查安装是否成功：
```bash
python -c "import megatron; print(megatron.__path__)"
```

## 步骤5: 获取数据集和检查点

- 按照 `Pai-Megatron-Patch/examples/qwen2` 中的说明下载检查点和数据集

- 将检查点、数据集文件夹和脚本 `run_qwen2_train.sh` 移动到 Pai-Megatron-Patch 目录。现在你的目录结构应该如下：
```
    /workspace/
    ├─ Pai-Megatron-Patch/
    │  ├─ qwen-ckpts/
    │  ├─ qwen-datasets/
    |  ├─ Pai-Megatron-Sophgo.patch
    │  ├─ run_qwen2_train.sh
    │  └─ ...
    ├─ ...
```

## 步骤6: 运行训练脚本

- 设置超参数并运行预训练脚本

参数说明：完整模型大小、层数（0 表示完整模型）、张量并行度(TP)和流水线并行度(PP)。

要使用 TP=2 预训练 7B 模型，运行：

```bash
source run_qwen2_train.sh 7B 0 2 1
```

要运行较小的测试，例如使用 TP=2 的单层 7B 模型，运行：

```bash
source run_qwen2_train.sh 7B 1 2 1
```

## 常见问题

- 确保没有安装 transformer_engine、apex 和 megatron-core。如果已安装，请卸载它们：

```bash
python -m pip uninstall transformer_engine apex megatron-core -y
```