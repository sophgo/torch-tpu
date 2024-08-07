import os
import sys
import shutil
import json
import torch
import torch_tpu
import numpy as np

from torch_tpu.utils.apply_revert_patch import apply_table, check_is_applied, check_version
is_ok = True
for package, version, _ in apply_table:
    is_ok &= check_version(package, version)
    is_ok &= check_is_applied(package)
if not is_ok:
    print("Error: check_version or check_is_applied failed")
    sys.exit(255)

os.environ["CMODEL_FAST_EXEC"]="1"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="60000000000"

def case1():
    os.chdir(torch_tpu.__path__[0] + "/demo/LLaMA-2_LoRA_Finetune")
    tmp_yaml_fn = "llama_lora_full_test.yaml"
    with open(tmp_yaml_fn, "w") as f:
        f.write("""### model
model_name_or_path: "meta-llama/Llama-2-7b-chat-hf"

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
    os.system(f"llamafactory-cli train {tmp_yaml_fn}")
    log_file = "output_dir_tpu/trainer_log.jsonl"
    losses = []
    for line in open(log_file):
        if "loss" in line:
            losses.append(json.loads(line)["loss"])
    losses_expected = [1.3193, 2.1143, 1.4869, 1.442, 1.2337, 1.1222, 1.0713, 1.1996, 0.8287, 0.9138] # cuda losses
    pass_test = np.allclose(np.array(losses), np.array(losses_expected), atol=1e-3, rtol=1e-3)
    print(f"losses: {losses}")
    print(f"losses_expected: {losses_expected}")
    if not pass_test:
        print(f"Test failed.")
    else:
        print(f"Test passed.")
    os.remove(tmp_yaml_fn)
    shutil.rmtree("output_dir_tpu")
    return pass_test

if __name__ == "__main__":
    # This test takes about 2 hours on a i7-12700 CPU
    ret = case1()
    if not ret:
        sys.exit(255)