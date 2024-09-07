分布式训练
============

本节介绍对分布式训练的支持。

TORCH-TPU支持DeepSpeed与Megatron-DeepSpeed框架的分布式训练。

支持分布式训练需要安装MegatronDeepSpeedTpu插件，该插件包含了DeepSpeed与Megatron-DeepSpeed框架的分布式训练的支持。

针对DeepSpeed，该插件支持DeepSpeed的Zero-1和Zero-2 CPU-Offload优化策略。

针对Megatron-DeepSpeed，该插件支持TensorParallel优化策略。

该插件还提供了bert与gpt模型的分布式训练的示例，用户可以参考这些示例来使用分布式训练。

插件的安装及使用请参考MegatronDeepSpeedTpu文件夹下的安装文档。
