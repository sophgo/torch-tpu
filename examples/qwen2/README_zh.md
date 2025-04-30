[English](./README.md) | [中文](./README_zh.md)

# 使用 PAI-Megatron-Patch 预训练 Qwen2-7B

## 步骤1: 设置基础环境

### 可选方式1: 检查 Linux 系统环境
确保系统中的 IOMMU（输入输出内存管理单元）服务已设置为 translated 模式。

你可以通过运行以下命令进行验证：`sudo dmesg | grep -i iommu`<br>
如果输出显示 IOMMU 类型为 `Translated`，则说明你的环境配置正确。
否则，请更新系统配置，将 IOMMU 设置为 translated 模式。

### 可选方式2: 准备 torch-tpu 环境

你可以参考[README.md]或者用户手册去设置torch_tpu环境。


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
git submodule update
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

## 步骤4: 安装依赖包

获取 PAI-Megatron-Patch 后，需要运行以下命令安装运行所需的依赖包`：

```bash
pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets netifaces nnmoduletools>=0.1.1 evaluate sacrebleu scikit-learn sqlitedict peft==0.10.0 pytablewriter
# pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets netifaces nnmoduletools>=0.1.1 evaluate sacrebleu scikit-learn sqlitedict peft==0.10.0 pytablewriter -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 步骤5: 获取数据集和检查点

- 按照 `Pai-Megatron-Patch/examples/qwen2` 中的说明下载数据集
- 从huggingface网站下载Qwen2-7B模型作为参考模型, 将`config.json`文件中`torch_dtype`参数修改为`float16`

- 将检查点、数据集文件夹、脚本 `run_qwen2_train.sh`与脚本`run_qwen2_evaluation.sh` 移动到 Pai-Megatron-Patch 目录。现在你的目录结构应该如下：
```
    /workspace/
    ├─ Pai-Megatron-Patch/
    │  ├─ qwen-ckpts/
    |  |  ├─Qwen2-7B
    |  |  └─ ...
    │  ├─ qwen-datasets/
    |  ├─ Pai-Megatron-Sophgo.patch
    │  ├─ run_qwen2_train.sh
    |  ├─ run_qwen2_evaluation.sh
    │  └─ ...
    ├─ ...
```

## 步骤6: 运行训练脚本并保存checkpoints

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


## Step7: 将MegatronCore格式的checkpoints转为HuggingFace格式，并评估该模型

- 启动评估脚本环境

```bash
source run_qwen2_evaluation.sh
```

- 需要设置的参数为：将要保存的HuggingFace格式的checkpoints文件路径，训练保存的MegatronCore格式的checkpoints文件路径(默认)， 从HuggingFace下载的参考模型路径
- 均使用默认参数，并运行下面命令：

```bash
mcore_to_hg
```
- 根据`Pai-Megatron-Patch/examples/qwen2`的指示下载评估数据集，或者您也可以直接运行下面命令：

```bash
cd /workspace
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/ceval.tgz
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/evaluate.tgz
tar -xvzf ceval.tgz
tar -xvzf evaluate.tgz
```

- 需要设置的参数为：保存的Huggingface格式的checkpoints文件路径
- 均使用默认参数，并运行下面命令：

```bash
evaluate
```


## 常见问题

- 确保没有安装 transformer_engine、apex 和 megatron-core。如果已安装，请卸载它们：

```bash
python -m pip uninstall transformer_engine apex megatron-core -y
```

- 运行脚本报错`ModuleNotFoundError: No module named 'megatron'`:

示例脚本中已经包含了添加`PYTHONPATH`的命令；如果仍然提示找不到`megatron`，可以尝试在`Pai-Megatron-Patch`目录下再次运行以下命令:
```bash
export PYTHONPATH=$(pwd)/PAI-Megatron-LM-240718:$PYTHONPATH
```
