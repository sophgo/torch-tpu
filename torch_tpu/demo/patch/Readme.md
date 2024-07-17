# SOPHGO高性能TPU BM1690对开源库的适配补丁

1. transformers 
主要修改`transformers`仓库trainer的逻辑，添加了对`torch-tpu`的支持。 

源仓库commit: ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2 (branch v4.41.2)

适配补丁: transformer-Sophgo.patch 

2. accelerate

主要修改`accelerate`仓库xxxx的逻辑，添加了对`torch-tpu`的支持。 

源仓库commit: b52803dc6f8d423cd9758cdd6f77ebbd4acba035  (branch v0.30.1)

适配补丁: accelerate-Sophgo.patch

3. llama factory

主要修改`accelerate`仓库xxxx的逻辑，添加了对`torch-tpu`的支持。 

源仓库commit: 

适配补丁: 

