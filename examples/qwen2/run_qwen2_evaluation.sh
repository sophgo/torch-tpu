#!/bin/bash

if [[ $PYTHONPATH != *PAI-Megatron-LM-240718* ]]; then
    export PYTHONPATH=$(pwd)/PAI-Megatron-LM-240718:$PYTHONPATH
fi

MEGATRON_PATH=$(pwd)

function mcore_to_hg()
{
    DEFAULT_TARGET_HG_MODEL="${MEGATRON_PATH}/qwen-ckpts/Qwen2-7B-mcore-to-hg-tp2-pp1"
    TARGET_HG_MODEL=${1:-$DEFAULT_TARGET_HG_MODEL}

    DEFAULT_CKPTS_MODEL="${MEGATRON_PATH}/qwen-ckpts/Qwen2-7B-ckpts"
    CKPTS_MODEL=${2:-$DEFAULT_CKPTS_MODEL}

    DEFAULT_REF_HG_MODEL="${MEGATRON_PATH}/qwen-ckpts/Qwen2-7B"
    REF_HG_MODEL=${3:-$DEFAULT_REF_HG_MODEL}

    rm -rf "${TARGET_HG_MODEL}"
    mkdir -p "${TARGET_HG_MODEL}"

    pushd "${MEGATRON_PATH}/toolkits/model_checkpoints_convertor/qwen" > /dev/null

    QWEN2_WHOLE_NET_TRANS=1 DISABLE_CACHE=1 trust_remote_code=True bash hf2mcore_qwen2_convertor.sh \
        7B \
        "${CKPTS_MODEL}" \
        "${TARGET_HG_MODEL}" \
        2 \
        1 \
        1 \
        fp16 \
        false \
        true \
        "${REF_HG_MODEL}"

    popd > /dev/null
}


function evaluate()
{
    DEFAULT_TARGET_HG_MODEL="${MEGATRON_PATH}/qwen-ckpts/Qwen2-7B-mcore-to-hg-tp2-pp1"
    TARGET_HG_MODEL=${1:-$DEFAULT_TARGET_HG_MODEL}

    pushd "${MEGATRON_PATH}/LM-Evaluation-Harness-240310" > /dev/null

    DISABLE_CACHE=1 python -m lm_eval \
    --model hf --model_args pretrained="${TARGET_HG_MODEL}" \
    --tasks ceval-valid  \
    --batch_size 8 \
    --device "tpu"

    popd > /dev/null
}