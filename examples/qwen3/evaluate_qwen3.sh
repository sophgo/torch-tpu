#!/bin/bash

echo "installing dependencies..."
uv pip install rouge_score==0.1.2

echo "evaluating..."
source /usr/src/server/soph_envsetup.sh

cp /workspace/tgi_evaluate_qwen3.py /usr/src/server/tests/soph_test/

python /usr/src/server/tests/soph_test/tgi_evaluate_qwen3.py \
--model-id /workspace/Qwen3-8B-sft-hf \
--dtype bfloat16 \
--dataset-path /workspace/qwen3-datasets/math_instruct_eval.json \
--output-path /workspace/results.json \
--batch-size 128 \
--max-new-tokens 512