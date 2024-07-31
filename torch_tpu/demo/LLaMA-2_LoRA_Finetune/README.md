# 示例：使用SOPHGO高性能TPU BM1690通过LLaMA-Factory仓库进行LLaMA-2 LoRA微调

## 1. 环境准备

### 1.1 安装torch-tpu

```bash
python -m pip install torch-tpu.XXXX.whl
```

### 1.2 从源码安装accelerate仓库并应用对应的SOPHGO补丁

```bash
git clone https://github.com/huggingface/accelerate
pushd accelerate
git checkout v0.30.1
git apply ${TORCH_TPU_HOME}/demo/patch/accelerate-Sophgo.patch
python -m pip install -e .
popd
```

### 1.3 从源码安装transformers仓库并应用对应的SOPHGO补丁

```bash
git clone https://github.com/huggingface/transformers
pushd transformers
git checkout v4.41.2
git apply ${TORCH_TPU_HOME}/demo/patch/transformers-Sophgo.patch
python -m pip install -e .
popd
```

### 1.4 从源码安装LLaMA-Factory仓库和依赖包并应用对应的SOPHGO补丁

```bash
git clone https://github.com/hiyouga/LLaMA-Factory/
pushd LLaMA-Factory
git checkout v0.8.3
git apply ${TORCH_TPU_HOME}/demo/patch/LLaMA-Factory-Sophgo.patch
python -m pip install -r requirements.txt
python -m pip install -e .
popd
```


## 2. 训练脚本

示例中已经包含了一个训练配置文件`llama_2_lora.yaml`，可以在`LLaMA-Factory`目录下直接使用。

```bash
cd LLaMA-Factory
llamafactory-cli train ${TORCH_TPU_HOME}/demo/LLaMA-2_LoRA_Finetune/llama_2_lora.yaml
```
