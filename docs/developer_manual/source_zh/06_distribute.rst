分布式的支持
============

本节介绍对分布式训练的支持。

TORCH-TPU支持DeepSpeed与Megatron-DeepSpeed框架的分布式训练。
支持分布式训练需要安装MegatronDeepSpeedTpu插件，该插件包含了DeepSpeed与Megatron-DeepSpeed框架的分布式训练的支持。
针对DeepSpeed，该插件支持DeepSpeed的Zero-1和Zero-2 CPU-Offload优化策略。
针对Megatron-DeepSpeed，该插件支持TensorParallel优化策略。
该插件还提供了bert与gpt模型的分布式训练的示例，用户可以参考这些示例来使用分布式训练。

安装插件请参考MegatronDeepSpeedTpu文件夹下的安装文档。
----


.. Lowering将Top层OP下沉到Tpu层OP, 它支持的类型有F32/F16/BF16/INT8对称/INT8非对称。

.. 当转换INT8时, 它涉及到量化算法; 针对不同硬件, 量化算法是不一样的, 比如有的支持perchannel, 有的不支持;

.. 有的支持32位Multiplier, 有的只支持8位, 等等。

.. 所以Lowering将算子从硬件无关层(TOP), 转换到了硬件相关层(TPU)。

.. 基本过程
.. ------------

.. .. _lowering:
.. .. figure:: ../assets/lowering.png
..    :height: 5cm
..    :align: center

..    Lowering过程

.. Lowering的过程, 如图所示(:ref:`lowering`)

.. * Top算子可以分f32和int8两种, 前者是大多数网络的情况; 后者是如tflite等量化过的网络的情况
.. * f32算子可以直接转换成f32/f16/bf16的tpu层算子, 如果要转int8, 则需要类型是calibrated_type
.. * int8算子只能直接转换成tpu层int8算子

.. 混合精度
.. ------------

.. .. _mix_prec:
.. .. figure:: ../assets/mix_prec.png
..    :height: 8cm
..    :align: center

..    混合精度

.. 当OP之间的类型不一致时, 则插入CastOp, 如图所示(:ref:`mix_prec`)。

.. 这里假定输出的类型与输入的类型相同, 如果不同则需要特殊处理, 比如embedding无论输出是什么类型, 输入都是uint类型。
