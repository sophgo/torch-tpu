混合精度训练
============

本节关于TORCH-TPU混精的支持。

由于GPU支持不同的数据类型(FP16、FP8、FP32等), 不同的精度对应不同的计算算力和数据搬移量，

低精度下算力往往会高很多，这样会加速计算的过程。PyTorch本身通过AMP模块实现了GPU对于混合精度（FP16、FP32）训练的支持。

TORCH-TPU同样支持了AMP模块，该部分代码位于torch_tpu的tpu/amp模块下，完全由python实现。

torch_tpu.amp实现了同cuda.amp同样的计算逻辑和接口。

相关使用可以参照https://pytorch.org/docs/stable/amp.html。