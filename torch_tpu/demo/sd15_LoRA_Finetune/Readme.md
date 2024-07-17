# 示例：使用SOPHGO高性能TPU BM1690进行Stable Diffusion LoRA微调


## 1. 环境准备

### 1.1 安装torch-tpu

```bash
python -m pip install torch-tpu.XXXX.whl
```

### 1.2 从源码安装accelerate仓库并应用对应的SOPHGO补丁

```bash
git clone https://github.com/huggingface/accelerate
cd accelerate
git checkout v0.30.1
git apply ${TORCH_TPU_HOME}/torch_tpu/demo/patch/accelerate-Sophgo.patch
python -m pip install -e .
```

### 1.3 安装其他依赖

```bash
pip3 install -r requirements.txt
```

## 2. 训练脚本

在当前目录下，有来自`https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py`,针对SOPHGO TPU做了适配。 

可以通过修改`train.sh`中模型路径和数据集路径进行训练。

训练更多配置信息请参考`train_text_to_image_lora.py`脚本。