[English](./README.md) | [中文](./README_zh.md)

# 使用 PAI-Megatron-Patch 微调 Qwen3-8B

## 步骤 1: 设置基础环境

### 1.1 检查 Linux 系统环境
确保系统中的 IOMMU（输入输出内存管理单元）服务已设置为 `translated` 模式。

您可以通过运行以下命令进行验证：
```bash
sudo dmesg | grep -i iommu
```
如果输出显示 IOMMU 类型为 **Translated**，则说明您的环境配置正确。否则，请更新系统配置，将 IOMMU 设置为 `translated` 模式。

### 1.2 准备 torch-tpu 环境

您可以参考根目录的 [README.md] 或用户手册 [dist/docs/TORCH_TPU快速入门指南.pdf] 来设置 `torch_tpu` 环境。


## 步骤 2: 在 Docker 容器中安装 torch_tpu
> **重要提示**：我们强烈推荐使用"方式 2"获取最新的 `torch_tpu` 安装包，以避免"方式 1"可能带来的运行失败或性能问题。

### 方式 1: 通过 pip 安装 torch_tpu

```bash
pip install torch_tpu
```

### 方式 2: 直接从 Release 链接安装 torch_tpu（推荐）
从 FTP 服务器上 torch_tpu/release_build/latest_release 目录下拉取 torch-tpu whl 包并安装：
```
tar -xvf torch-tpu_*.tar.gz
pip install dist/torch_tpu-*_x86_64.whl --force-reinstall
```

## 步骤 3: 获取 PAI-Megatron-Patch

### 3.1 克隆仓库

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git /workspace/Pai-Megatron-Patch
cd /workspace/Pai-Megatron-Patch
# 切换到指定版本
git checkout b9fd9c2
git submodule update
```

### 3.2 应用补丁

```bash
# 假设您当前在 ../examples/qwen3 示例目录下
cp Pai-Megatron-Sophgo-Qwen3.patch /workspace/Pai-Megatron-Patch/
cd /workspace/Pai-Megatron-Patch
git apply Pai-Megatron-Sophgo-Qwen3.patch
```

## 步骤 4: 安装依赖包

获取 PAI-Megatron-Patch 后，需要运行以下命令安装运行所需的依赖包`：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 步骤 5: 获取基模型与数据集

### 5.1 下载模型

从 [Hugging Face](https://huggingface.co/) 网站下载 `Qwen3-8B-Base` 模型作为基础模型。

### 5.2 下载数据集

执行以下命令下载训练数据集：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
cd /workspace
python3 -m dfss --url=open@sophgo.com:SC11-FP300/LLM-finetune/qwen3-datasets/qwen3-datasets.zip
unzip qwen3-datasets.zip
```

### 5.3 复制执行脚本

将当前目录下的执行脚本拷贝到工作路径：

```bash
# 在 ../examples/qwen3 示例目录下执行
cp *.sh *.py /workspace/
```

### 5.4 最终文件目录结构

完成上述步骤后，文件目录结构应如下所示：

```
/workspace/
├─ Pai-Megatron-Patch/          # PAI-Megatron-Patch 代码仓库
│  └─ Pai-Megatron-Sophgo-Qwen3.patch
├─ qwen3-datasets/              # 训练集与验证集
├─ Qwen3-8B-Base/               # HuggingFace 格式的基础模型
├─ run_qwen3_train.sh           # 训练脚本
├─ convert_qwen3_model.sh       # 模型转换脚本
├─ evalute_qwen3.sh             # 模型评估脚本
├─ tgi_evaluate_qwen3.py        # TGI 评估 Python 脚本
└─ ...
```

## 步骤 6: 进行微调训练

### 6.1 转换模型格式（HuggingFace → MCore）

在训练之前，需要将 HuggingFace 格式的模型转换为 MCore 格式：

```bash
source /workspace/convert_qwen3_model.sh
hf2mcore
```

### 6.2 执行微调训练

训练脚本默认执行 3000 次迭代（约 2 个 epoch），每 1000 次迭代保存一次 checkpoint：

```bash
bash /workspace/run_qwen3_train.sh
```

### 6.3 转换模型格式（MCore → HuggingFace）

训练完成后，将 MCore 格式的模型转换回 HuggingFace 格式以便后续使用：

```bash
source /workspace/convert_qwen3_model.sh
mcore2hf
```

### 6.4 训练后的文件目录

完成训练后，将生成以下新的目录：

```
/workspace/
├─ Pai-Megatron-Patch/
├─ ...                          # 同上，省略
├─ Qwen3-8B-to-mcore-tp2        # MCore 格式的基础模型（TP=2）
├─ Qwen3-8B-sft-mcore           # 训练过程中保存的 checkpoint 模型
├─ Qwen3-8B-sft-hf              # 微调后的模型（HuggingFace 格式）
└─ ...
```

## 步骤 7: 评估模型

您可以根据需求选择任意推理框架对模型进行评估。本示例使用 **TGI-TPU**（Text Generation Inference for Sophgo TPU）进行推理，并使用 **ROUGE 分数**作为评估指标。

### 7.1 环境搭建

详细内容参考 FTP 服务器上 `LLMs/text-generation-inference/release_build/latest_release` 目录下的 `text-generation-inference_quick_start_zh.pdf` 文档搭建 TGI-TPU Docker 环境。

仅需拉取对应的 Docker 镜像，创建并进入容器即可，参考命令如下：

```bash
# 创建容器
docker run --privileged -itd --restart always \
  --name <CONTAINER_NAME> \
  --shm-size 1g \
  -p 8080:80 \
  -v $(pwd):/workspace \
  -v /dev/:/dev/ \
  -v /opt/tpuv7:/opt/tpuv7 \
  --entrypoint /bin/bash \
  soph_tgi:3.2.0-slim

# 进入容器
docker exec -it <CONTAINER_NAME> bash
```

### 7.2 运行推理与评估

在 TGI-TPU 容器中运行以下命令进行推理和评估：

```bash
bash /workspace/evaluate_qwen3.sh 
```

该脚本将：
- 启动 TGI-TPU 推理服务
- 使用微调后的模型进行推理
- 计算 ROUGE 分数并输出评估结果表格
- 将推理结果保存为 `/workspace/results.json`文件
- 最终输出ROUGE分数应与下述相似：
```
==========================================================================================
                                 ROUGE evaluation results                                 
==========================================================================================
Metric       Recall                    Precision                 F-Measure                
------------------------------------------------------------------------------------------
ROUGE-1      0.5625 (±0.2373)          0.5997 (±0.2921)          0.5296 (±0.2446)         
ROUGE-2      0.3621 (±0.2358)          0.3867 (±0.2571)          0.3433 (±0.2320)         
ROUGE-L      0.4739 (±0.2353)          0.5004 (±0.2696)          0.4435 (±0.2345)         
==========================================================================================
```

## 常见问题

### Q1: 依赖包冲突问题

确保没有安装 `transformer_engine`、`apex` 和 `megatron-core`。如果已安装，请先卸载：

```bash
python -m pip uninstall transformer_engine apex megatron-core -y
```

### Q2: 找不到 megatron 模块

**错误信息**：`ModuleNotFoundError: No module named 'megatron'`

**解决方法**：示例脚本中已包含添加 `PYTHONPATH` 的命令。如果仍然提示找不到 `megatron`，请手动运行：

```bash
export PYTHONPATH=/workspace/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624:$PYTHONPATH
```

### Q3: TGI-TPU 推理报错

如果在使用 TGI-TPU 推理时遇到错误，请首先参考 `text-generation-inference_quick_start_zh.pdf` 文档的 4.2.1 章节测试环境是否安装成功，以排除 TGI-TPU 环境搭建问题。

### Q4: 如何使用指定芯片

例如您有8个芯片，你想指定使用最后2个芯片，即chip6与chip7，您可以在命令前加上环境变量`CHIP_MAP=6,7`，该环境变量会指定分布式训推需要使用的芯片编号。

---

如有其他问题，欢迎提交 Issue 或联系技术支持团队。