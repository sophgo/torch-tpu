import os
import sys
import shutil
import json
import torch
import torch_tpu
import numpy as np
from torch_tpu.utils.apply_revert_patch import apply_table, check_is_applied, check_version
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

os.environ["CMODEL_FAST_EXEC"]="1"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="60000000000"

def patch_datasets():
    import datasets
    fn = datasets.__path__[0] + "/utils/py_utils.py"
    with open(fn, "a") as f:
        f.write("\n")
        f.write("has_sufficient_disk_space = lambda needed_bytes, directory='.': True\n")

def check_requirements():
    print("Checking requirements...", flush=True)
    is_ok = True
    for package, version, _ in apply_table:
        is_ok &= check_version(package, version)
        is_ok &= check_is_applied(package)
    if not is_ok:
        print("Error: check_version or check_is_applied failed", flush=True)
    else:
        print("Requirements are satisfied.", flush=True)
    return is_ok

def gen_model(model_fn):
    print("Downloading tokenizer...", flush=True)
    os.mkdir(model_fn)
    os.system(f"wget https://hf-mirror.com/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model -O {model_fn}/tokenizer.model")
    os.system(f"wget https://hf-mirror.com/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer_config.json -O {model_fn}/tokenizer_config.json")
    print("Generating fake llama-2-7b model...", flush=True)
    llama_2_7b_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "use_cache": True,
        "vocab_size": 32000}
    config = LlamaConfig(**llama_2_7b_config)
    llama_model = LlamaForCausalLM(config).half()
    llama_model.save_pretrained(model_fn)
    print("Model and tokenizer are generated.", flush=True)

def gen_input(model_fn, tmp_yaml_fn):
    print("Generating yaml file...", flush=True)
    with open(tmp_yaml_fn, "w") as f:
        f.write(f"""### model
model_name_or_path: "{model_fn}"

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj

### dataset
dataset: alpaca_gpt4_zh
template: llama2
cutoff_len: 1024
max_samples: 10
overwrite_cache: false
preprocessing_num_workers: 64

### output
output_dir: "output_dir_tpu"
logging_steps: 1
save_steps: 10
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
disable_gradient_checkpointing: true
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
fp16: true""")
    print("Yaml file is generated.", flush=True)
    print("Downloading dataset...", flush=True)
    os.system("wget https://hf-mirror.com/datasets/llamafactory/alpaca_gpt4_zh/resolve/main/alpaca_gpt4_data_zh.json -O data/alpaca_gpt4_data_zh.json")
    with open("data/dataset_info.json", "w") as f:
        f.write("""{"alpaca_gpt4_zh": {"file_name": "alpaca_gpt4_data_zh.json"}}""")

def case1():
    os.chdir(torch_tpu.__path__[0] + "/demo/LLaMA-2_LoRA_Finetune")
    model_fn = "llama_7b_fake"
    tmp_yaml_fn = "llama_lora_full_test.yaml"
    gen_model(model_fn)
    gen_input(model_fn, tmp_yaml_fn)
    print(f"Run LLaMA-Factory LoRA Test", flush=True)
    os.system(f"HF_DATASETS_CACHE=`realpath ./{model_fn}` HF_ENDPOINT=https://hf-mirror.com llamafactory-cli train {tmp_yaml_fn}")
    log_file = "output_dir_tpu/trainer_log.jsonl"
    losses = []
    for line in open(log_file):
        if "loss" in line:
            losses.append(json.loads(line)["loss"])
    losses_expected = [11.1109, 11.1008, 11.3482, 11.1663, 11.1816, 11.2014, 11.1593, 10.957, 11.1613, 11.2404] # cuda losses for seed 2260
    pass_test = np.allclose(np.array(losses), np.array(losses_expected), atol=1e-3, rtol=1e-3)
    print(f"losses_actual : {losses}", flush=True)
    print(f"losses_desired: {losses_expected}", flush=True)
    if not pass_test:
        print(f"Test failed.", flush=True)
    else:
        print(f"Test passed.", flush=True)
    os.remove(tmp_yaml_fn)
    shutil.rmtree(model_fn)
    shutil.rmtree("output_dir_tpu")
    return pass_test

if __name__ == "__main__":
    ret = check_requirements()
    if not ret:
        sys.exit(255)
    patch_datasets()
    torch.manual_seed(2260)
    ret = case1()
    if not ret:
        sys.exit(255)