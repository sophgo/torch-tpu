#!/bin/bash
set -ex
set -o pipefail

BASE_PATH=dataset
DATA_PATH=${BASE_PATH}/gpt_training_data_text_document
DS_CONFIG=ds_config.json

TP=8
PP=1
NLAYERS=40
HIDDEN=5120
NHEADS=40
SEQ_LEN=2048
POS_EMB=2048

GLOBAL_BATCH=2
MICRO_BATCH=2

ZERO_STAGE=2

OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "offload_optimizer": {
      "device": "cpu"
    }
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


python -u pretrain_gpt.py \
    --local_rank $LOCAL_RANK \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $POS_EMB \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 10 \
    --lr 1.e-4 \
    --log-interval 1 \
    --data-path $DATA_PATH \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $BASE_PATH/gpt2-vocab.json \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --save-interval 0 \
    --eval-iters 0 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --fp16 \
    --tensorboard-dir $OUTPUT_DIR \
    --use-cpu-initialization \
    --cpu-optimizer \
    --no-gradient-accumulation-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --checkpoint-activations \
    $ds_args 2>&1 | tee ${OUTPUT_DIR}/output_${RANK}.log

    # --attention-dropout 0.0 \
    # --hidden-dropout 0.0 \
