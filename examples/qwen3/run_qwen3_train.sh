# Pai-Megatron-Patch
set -x

########################################################
# 后续传入的参数列表如下
########################################################
# ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
# MODEL_SIZE=$2                   # 模型结构参数量级: 0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B
# BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
# GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
# LR=$5                           # 学习率
# MIN_LR=$6                       # 最小学习率
# SEQ_LEN=$7                      # 序列长度
# PAD_LEN=$8                      # Padding长度
# PR=${9}                         # 训练精度: fp16, bf16, fp8
# TP=${10}                        # 模型并行度
# PP=${11}                        # 流水并行度
# CP=${12}                        # 上下文并行度
# ETP=${13}                       # 专家张量并行度
# EP=${14}                        # 专家模型并行度
# SP=${15}                        # 是否使用序列并行: true, false
# DO=${16}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
# FL=${17}                        # 是否优先使用Flash Attention: true, false
# SFT=${18}                       # 是否执行微调训练: true, false
# AC=${19}                        # 激活检查点模式: sel, full, offload, false
# OPTIMIZER_OFFLOAD=${20}         # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
# SAVE_INTERVAL=${21}             # 保存ckpt的间隔
# DATASET_PATH=${22}              # 训练数据集路径
# VALID_DATASET_PATH=${23}        # 验证数据集路径
# PRETRAIN_CHECKPOINT_PATH=${24}  # 预训练模型路径
# TRAIN_TOKENS_OR_ITERS=${25}     # 训练TOKEN或者Iter数
# WARMUP_TOKENS_OR_ITERS=${26}    # 预热TOKEN或者Iter数        
# OUTPUT_BASEPATH=${27}           # 训练输出日志文件路径

########################################################
# 执行训练
########################################################
pushd /workspace/Pai-Megatron-Patch/examples/qwen3

WHOLE_NET_TRANS=1 DISABLE_CACHE=1 TPU_ALLOCATOR_FREE_DELAY_IN_MS=100 sh run_mcore_qwen3.sh   \
dsw  \
8B   \
8    \
32 \
1e-5  \
1e-6 \
512  \
512  \
bf16  \
2   \
1  \
1 \
1 \
1 \
true \
false   \
false \
true \
false   \
false \
1000  \
/workspace/qwen3-datasets/math_instruct_train_text_document \
/workspace/qwen3-datasets/math_instruct_train_text_document  \
/workspace/Qwen3-8B-to-mcore-tp2 \
3000  \
100  \
/workspace/Qwen3-8B-sft-mcore

popd > /dev/null

set +x
