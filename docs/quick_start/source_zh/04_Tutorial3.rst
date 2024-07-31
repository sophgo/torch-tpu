===================
模型训练实例_BM1690_LLaMA-2_LoRA_Finetune
===================

本章节介绍了如何在使用Torch-TPU的前提下，使用LoRA进行LLaMA-2的Finetune的一个示例。

本节相关背景已经在上一章节中介绍过，本节假设用户已经安装torch-tpu环境。

如何获取Demo代码
==================

Demo代码随着torch-tpu的whl包一起发布，用户可以在torch-tpu的whl包中找到 "demo/LLaMA-2_LoRA_Finetune" 代码。具体可以安装如下方式查找torch-tpu的whl包的安装路径。 

.. code-block :: bash

    pip show torch-tpu


接下来，将"TORCH_TPU_HOME"指代"torch-tpu"的路径。

训练流程可以参考 "demo/LLaMA-2_LoRA_Finetune" 中的 README.md 文件，与本节下文相同。

开始前的准备
==================

下载安装 Accelerate
------------------

Torch_tpu 对Accelerate仓库添加了补丁，以支持TPU的训练。补丁位置位于"demo/patch"。

用户需要从github上下载Accelerate的代码，并使用Torch_tpu提供的补丁，随后再进行安装。

具体指令如下：

.. code-block :: bash

    git clone https://github.com/huggingface/accelerate
    pushd accelerate
    git checkout v0.30.1
    git apply ${TORCH_TPU_HOME}/demo/patch/accelerate-Sophgo.patch
    python -m pip install -e .
    popd


下载安装 transformers
------------------

对 transformers 仓库，也需要进行类似的操作。

.. code-block :: bash

    git clone https://github.com/huggingface/transformers
    pushd transformers
    git checkout v4.41.2
    git apply ${TORCH_TPU_HOME}/demo/patch/transformers-Sophgo.patch
    python -m pip install -e .
    popd


下载安装 LLaMA-Factory
------------------

对 LLaMA-Factory 仓库，也需要进行类似的操作，还需要额外安装依赖包。

.. code-block :: bash

    git clone https://github.com/hiyouga/LLaMA-Factory/
    pushd LLaMA-Factory
    git checkout v0.8.3
    git apply ${TORCH_TPU_HOME}/demo/patch/LLaMA-Factory-Sophgo.patch
    python -m pip install -r requirements.txt
    python -m pip install -e .
    popd


开始训练
==================

进入 "${TORCH_TPU_HOME}/demo/LLaMA-2_LoRA_Finetune" 目录，根据实际情况修改 "llama_2_lora.yaml" 文件中的模型位置。
随后进入"LLaMA-Factory"安装目录，执行 "llamafactory-cli train ${TORCH_TPU_HOME}/demo/LLaMA-2_LoRA_Finetune/llama_2_lora.yaml" 即可开始训练。训练更多配置信息请参考官网其他示例。