# MegatronDeepSpeedTpu

## 简介

本项目致力于开发一款为算能深度学习处理器TPU量身定制的人工智能编程框架PyTorch的插件，以支持深度学习分布式训练推理框架`DeepSpeed`及`Megatron-DeepSpeed`在算能深度学习处理器BM1684X与BM1690上的无缝适配。通过这种方式，开发者可以在一个熟悉的环境中继续他们的研究和开发工作，同时借助算能硬件的强大加速能力和分布式训练框架的优化功能，大幅提升工作效率。

本项目旨在为算能深度学习处理器上工作的PyTorch用户提供一个既实用又高效的工具，帮助他们充分利用这些先进硬件进行深度学习模型大规模分布式训练任务。开发者将能更便捷地在这些处理器上进行大规模模型的分布式训练，不仅提高了效率，也提升了训练过程的可扩展性。我们期望通过这些努力，为深度学习及大模型技术的发展与应用贡献力量。

## 目录结构

本项目包含以下内容：

- `megatron_deepspeed_tpu`：包含插件的主要源代码。
    - `adaptor`：包括`torch`、`megatron`和`deepspeed`插件的源代码。通过monkey-patch技术实现，文件名与原始框架的对应文件名相匹配。
    - `debugger`：为开发者提供的可嵌入框架或模型的调试代码。
- `tpu_workarounds`：包含一系列魔法函数，用于适配TPU功能，解决TPU与GPU设备功能表现的差异。
- `examples`：展示如何使用本插件在BM1690虚拟仿真环境(CMODEL模式)上进行训练的示例程序集。
    - `bert`：展示如何使用DeepSpeed框架进行分布式BERT网络训练的示例程序。参考源[HelloDeepSpeed](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/HelloDeepSpeed)
    - `gpt`：展示如何使用Megatron-DeepSpeed框架进行分布式GPT网络训练的示例程序。参考源[examples_deepspeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples_deepspeed/)

## 功能

[DeepSpeed](https://github.com/microsoft/DeepSpeed)是由微软研究院推出的开源深度学习优化库，旨在加速大规模模型的训练速度和提高效率。该框架专为分布式训练环境设计，能够在多主机多设备条件下上高效运行超大规模模型。通过采用创新的技术ZeRO(Zero Redundancy Optimizer)优化器，`DeepSpeed`显著减少了模型训练过程中设备的内存占用，使得在有限资源下可以训练更大的模型。`DeepSpeed`引入加速器(accelerator)概念，支持模型在除GPU以外的多种设备上运行，如CPU、NPU、XPU等。本项目实现了专门适配算能深度学习处理器的加速器`tpu_accelerator`，以便在算能TPU上使用`DeepSpeed`框架执行训练任务。

[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)是一个集成了[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)和`DeepSpeed`技术的高性能深度学习训练框架，旨在为研究人员和开发人员提供一种更加高效、可扩展的方式来训练大型语言模型。`Megatron-LM`是由NVIDIA开发的一个开源项目，专门用于训练大规模的语言模型。该项目通过模型并行化技术，支持数据并行(Data Parallelism)、张量并行(Tensor Parallelism)、流水线并行(Pipeline Parallelism)等多种模型并行方式，允许模型在多个GPU上高效地分布和训练，优化了大型模的训练过程。`Megatron-DeepSpeed`框架结合了这两个项目的优势，在GPU以外的多种计算设备上适配了多种模型并行技术，让训练大规模模型变得更加高效和可行。本项目将`Megatron-DeepSpeed`框架适配于算能深度学习处理器，以实现在算能TPU上使用`Megatron-DeepSpeed`框架执行训练任务。

本插件目前已经实现了一系列功能，旨在优化和加速深度学习模型的训练过程。具体支持的功能如下：

- **ZeRO优化的支持**：适配了`DeepSpeed`的ZeRO优化技术的Stage 1与Stage 2，可以实现模型的优化器或梯度分布在多个设备上，有效降低了模型训练过程中设备上的内存占用，从而实现更大模型的训练。

- **CPU Offloading**：结合使用`cpu_adam`，本插件支持将内存密集型的adam优化器的更新操作转移到CPU内存上进行。这一功能可以显著减少设备内存的使用，使得在有限的设备资源下也能训练大型模型。

- **模型并行性支持**：支持`Megatron-DeepSpeed`的张量并行或流水线并行功能，允许模型在多个设备和多个主机上进行分布式训练。张量并行是通过分割模型的参数到不同的设备上实现的，而流水线并行则是将模型的不同部分分布到多个设备上，每个设备负责模型的一部分计算。这两种并行技术可以显著提高训练效率和扩展模型的大小。

- **易于集成和使用**：尽管提供了强大的功能，但本插件仍然保持了易于集成和使用的特点。开发者只需简单地在现有代码中添加少许代码，即可享受到本插件带来的各项优势。

通过这些功能，本插件为深度学习研究者和开发者提供了一个强大而灵活的工具，以更高效、更低成本地训练大型和复杂的神经网络模型。

## 安装

为了确保在Docker环境中使用本插件时，能够进行多主机训练，请确保在执行`docker run`命令时，附加`--network host`选项，以实现Docker容器内外的通信。

本插件的安装步骤：

1. 参考`torch_tpu`的官方安装指南，完成`torch_tpu`与`sccl`的安装。
1. 安装`DeepSpeed v0.13.5`版本：

    ```shell
    git clone https://github.com/microsoft/DeepSpeed
    cd DeepSpeed
    git checkout v0.13.5
    pip install .
    ``` 

1. 安装`Megatron-DeepSpeed`：

    ```shell
    git clone https://github.com/microsoft/Megatron-DeepSpeed
    cd Megatron-DeepSpeed
    git checkout bcedecd1ff788d4d363f3365fd396053a08d65be
    pip install .
    ```

1. 安装`nvcc`：即使环境中不包含NVIDIA GPU也需要安装。

    ```shell
    sudo apt update
    sudo apt install nvidia-cuda-toolkit -y
    ```

1. 安装`apex`：

    ```shell
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
    ```

    如果需要使用CUDA版`apex`进行数据比对，可在装有CUDA的环境中运行以下命令安装`apex`：

    ```shell
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    ```

    请注意，`nvcc`和`apex`是`megatron`框架的依赖项。尽管适配后的框架不直接使用这两个库，但为避免安装和使用过程中出现错误，仍需进行安装。

    对于需要CUDA支持的环境，我们建议通过`conda`在独立的环境中进行安装，以避免与TPU环境冲突。

1. 安装依赖包:

    在本项目的根目录下执行

    ```shell
    pip install -r requirements.txt
    ```

1. 安装本插件`MegatronDeepSpeedTpu`:

    ```shell
    pip install .
    ```
    (或者 `pip install -e .` 命令安装开发者版本)

## 快速入门指南

本插件的使用非常便捷，仅需在现有训练脚本中加入以下单行代码：

import torch

```Python
import torch
import torch_tpu
import deepspeed
import megatron_deepspeed_tpu # <===== 此行即为所加
import megatron
```

当采用`Megatron-DeepSpeed`框架进行开发时，请确保在引入`megatron`库之前先导入本插件。

插件中的多个模块都可以利用环境变量来独立控制开关。当使用算能处理器TPU时，所有模块都默认打开，可通过`export DISABLE_ADAPTOR=1`或`export ENABLE_DEEPSPEED_ADAPTOR=0`等环境变量关闭。

开发者也可以参考[examples](examples/)中的示例程序在训练脚本中添加调试信息。

## 示例程序

本项目提供了`bert`和`gpt`两种网络训练的示例程序。详情请参考[examples](examples/)。



