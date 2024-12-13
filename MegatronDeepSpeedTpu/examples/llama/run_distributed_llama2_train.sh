#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex
set -o pipefail

######################################
# Change the below configurations here
BASE_PATH=../../../../Megatron-DeepSpeed/dataset
DS_CONFIG=ds_config.json
DATASET=${BASE_PATH}/llama_training_data_text_document
CHECKPOINT_PATH=.
TOKENIZER_PATH=tokenizer.model # offical llama tokenizer.model

TP=16
PP=1
ZERO_STAGE=2

HIDDEN_SIZE=8192 #4096 #  e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=28672 # 11008 # e.g. llama-13b: 13824
NUM_LAYERS=1 # 32 # e.g. llama-13b: 40
NUM_HEADS=64 # 32 # e.g. llama-13b: 40
SEQ_LENGTH=4096 # 2048 #  
NUM_KV_HEADS=8 # 32 # llama2 70B uses GQA

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=2 # e.g. llama: 4M tokens
TRAIN_STEPS=2 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################



cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 14
  },

  "wall_clock_breakdown" : true
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi


python -u pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 98,2,0 \
       --lr $LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 0 \
       --eval-iters 0 \
       --fp16 \
       --use-cpu-initialization \
       --cpu-optimizer \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --no-gradient-accumulation-fusion \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --num-key-value-heads $NUM_KV_HEADS \
       $ds_args
