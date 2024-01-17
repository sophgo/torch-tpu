使用 Sophgo BM1684x 实现 MNIST 图像分类               
=====================================

**摘要**:<br>
本文档提供了使用Torch-tpu在Minst数据集上进行训练的说明。包括使用`fp32`、`fp16`、`混合精度`三部分，分别对应三个脚本文件。

1.使用fp32精度进行训练
---
```
python mnist_fp32.py
```

2.使用fp16精度进行训练
---
```
python mnist_fp16.py
```

3.使用混合精度进行训练
---
```
python mnist_mix_precision.py
```