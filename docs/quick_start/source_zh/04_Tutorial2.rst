===================
模型训练实例_BM1690_StableDiffusion
===================

本章节介绍了如何在使用Torch-TPU的前提下，使用LoRA进行Stable-Diffusion的Finetune的一个示例。

本节相关背景已经在上一章节中介绍过，本节假设用户已经安装torch-tpu环境。

如何获取Demo代码
==================

Demo代码随着torch-tpu的whl包一起发布，用户可以在torch-tpu的whl包中找到 "demo/sd15_LoRA_Finetune" 代码。具体可以安装如下方式查找torch-tpu的whl包的安装路径。 

.. code-block :: bash

    pip show torch-tpu


接下来，将"TORCH_TPU_HOME"指代"torch-tpu"的路径。

训练流程可以参考 "demo/sd15_LoRA_Finetune" 中的 Readme.md 文件，与本节下文相同。

开始前的准备
==================

下载安装 Accelerate
------------------

Torch_tpu 对Accelerate仓库添加了补丁，以支持TPU的训练。补丁位置位于"demo/patch"。

用户需要从github上下载Accelerate的代码，并使用Torch_tpu提供的补丁，随后再进行安装。

具体指令如下：

.. code-block :: bash

    git clone https://github.com/huggingface/accelerate
    cd accelerate
    git checkout v0.30.1
    git apply ${TORCH_TPU_HOME}/torch_tpu/demo/patch/accelerate-Sophgo.patch
    pip install -e .


安装其他必要环境
------------------

.. code-block :: bash

    cd ${TORCH_TPU_HOME}/demo/sd15_LoRA_Finetune
    pip3 install -r requirements.txt


开始训练
==================

进入 "${TORCH_TPU_HOME}/demo/sd15_LoRA_Finetune" 目录，根据实际情况修改 "train.sh" 文件中的模型和数据集的位置，随后执行 "train.sh" 即可开始训练。训练更多配置信息请参考`train_text_to_image_lora.py`脚本。