#!/bin/bash

function hf2mcore()
{
    pushd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

    ########################################################
    # 执行转换
    # 下述命令将Qwen3-8B-Base模型转换为Qwen3-8B-to-mcore-tp2模型
    ########################################################
    USE_HOST_GLOO=1  DISABLE_CACHE=1 TPU_ALLOCATOR_FREE_DELAY_IN_MS=100 bash scripts/qwen3/run_8xH20.sh \
    8B \
    /workspace/Qwen3-8B-Base \
    /workspace/Qwen3-8B-to-mcore-tp2  \
    false \
    false \
    bf16 \

    popd
}


function mcore2hf()
{
    pushd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

    ########################################################
    # 执行转换
    # 下述命令将Qwen3-8B-sft-mcore模型转换为Qwen3-8B-sft-hf模型
    ########################################################

    USE_HOST_GLOO=1  DISABLE_CACHE=1 TPU_ALLOCATOR_FREE_DELAY_IN_MS=100 bash scripts/qwen3/run_8xH20.sh \
    8B \
    /workspace/Qwen3-8B-sft-mcore/checkpoint/finetune-mcore-qwen3-moe-megatron-8B-lr-1e-5-minlr-1e-6-bs-8-gbs-32-seqlen-512-pr-bf16-tp-2-pp-1-cp-1-ac-false-do-false-sp-true-ti-3000-wi-100 \
    /workspace/Qwen3-8B-sft-hf  \
    true \
    false \
    bf16 \
    /workspace/Qwen3-8B-Base

    popd
}