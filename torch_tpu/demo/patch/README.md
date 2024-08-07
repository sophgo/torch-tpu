# SOPHGO高性能TPU BM1690对开源库的适配补丁

1. transformers 
主要修改`transformers`仓库trainer的逻辑，添加了对`torch-tpu`的支持。 

源仓库commit: transformers_commit_id (tag: transformers_commit_tag)

适配补丁: transformer-Sophgo.patch 

2. accelerate

主要修改`accelerate`仓库GradScaler的逻辑，添加了对`torch-tpu`的支持。 

源仓库commit: accelerate_commit_id (tag: accelerate_commit_tag)

适配补丁: accelerate-Sophgo.patch

3. LLaMA-Factory

主要修改`LLaMA-Factory`仓库设备初始化的逻辑，添加了对`torch-tpu`的支持。 

源仓库commit: LLaMA-Factory_commit_id (tag: LLaMA-Factory_commit_tag)

适配补丁: LLaMA-Factory-Sophgo.patch

