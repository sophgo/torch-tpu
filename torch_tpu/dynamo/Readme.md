# 示例：使用SOPHGO高性能TPU BM1690通过编译模式进行Resnet50训练

## 1. 环境准备

### 1.1 安装torch-tpu

```bash
python -m pip install torch-tpu.XXXX.whl
```

### 1.2 安装 tpu-mlir

```bash
python -m pip install tpu-mlir.XXXX.whl
```

## 2. 训练

示例中已经包含了一个训练脚本`models/resnet50/joint_resnet_tpu.py`，可以在`dynamo`目录下使用.

```bash
cd torch_tpu/dynamo
export DISABLE_CACHE=1
python models/resnet50/joint_resnet_tpu.py
```

观察现象：loss 显著下降。

## 3. 可能问题及排查

如果观察到模型运行完后，卡住不动，需要检查是否开启了`DISABLE_CACHE`环境变量，如果没有开启，需要开启，以及是否关了`REMOVE_POLLS_IN_LLM`.