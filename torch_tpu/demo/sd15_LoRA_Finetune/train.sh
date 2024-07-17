# export CMODEL_FAST_EXEC=1
# export CMODEL_GLOBAL_MEM_SIZE=10000000000
export ACCELERATE_TORCH_DEVICE="tpu:0"
export MODEL_NAME=/workspace/llamatrain/stable-diffusion-v1-5
export DATASET_NAME="/workspace/llamatrain/test/all_in_one/dataset/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"

python sd15_lora_finetune.py \
--mixed_precision="fp16" \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_NAME --caption_column="text" \
--resolution=512 \
--train_batch_size=1 \
--num_train_epochs=4 --checkpointing_steps=50 \
--learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--output_dir="sd15_pokemon-model-lora_fp16" \
--validation_prompt="cute dragon creature" 