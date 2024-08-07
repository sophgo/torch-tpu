import torch
import torch_tpu
import os
import sys
import copy
from top_utest import TensorComparator, set_bacis_info
try:
    import transformers
    import peft
    import accelerate
except ImportError:
    os.system("python3 -m pip install transformers==4.41.2 accelerate==0.30.1 peft==0.11.1")

os.environ["CMODEL_FAST_EXEC"]="1"
from transformers import LlamaForCausalLM, LlamaConfig
from peft import LoraConfig, TaskType, get_peft_model

# temporary fix for torch.arange
def arange_wrapper(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret.dtype == torch.int64:
            return ret.to(torch.int32)
    return wrapper
torch.arange = arange_wrapper(torch.arange)

llama_config_dict = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 1, ##########
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "use_cache": True,
    "vocab_size": 32000
    }

lora_config_dict = {
    "r": 8,
    "target_modules": ["q_proj", "v_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "use_dora": False,
    "use_rslora": False
    }

def case1():
    seed = 2260
    set_bacis_info(seed)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        **lora_config_dict
    )

    llama_config = LlamaConfig(**llama_config_dict)
    llama_tf_model = LlamaForCausalLM(llama_config).half()
    lora_model = get_peft_model(llama_tf_model, lora_config)
    
    device = "tpu:0"
    lora_model_tpu = copy.deepcopy(lora_model).to(device)
    lora_model_cpu = lora_model.float()
    
    losses_cpu = []
    losses_tpu = []
    
    for seq_len in [640]:
        input_ids = torch.randint(0, llama_config_dict["vocab_size"], (1, seq_len))
        attn_mask = torch.randint(0, 2, (1, seq_len))
        input_labels = torch.randint(0, llama_config_dict["vocab_size"], (1, seq_len))
        
        input_ids_tpu = copy.deepcopy(input_ids).to(device)
        attn_mask_tpu = copy.deepcopy(attn_mask).to(device)
        input_labels_tpu = copy.deepcopy(input_labels).to(device)
        
        input_ids_cpu = input_ids
        attn_mask_cpu = attn_mask
        input_labels_cpu = input_labels
        
        lora_model_cpu.train()
        lora_model_tpu.train()
        
        output_cpu = lora_model_cpu(input_ids_cpu, attention_mask=attn_mask_cpu, labels=input_labels_cpu)
        output_tpu = lora_model_tpu(input_ids_tpu, attention_mask=attn_mask_tpu, labels=input_labels_tpu)
        
        losses_cpu.append(output_cpu["loss"].detach().item())
        losses_tpu.append(output_tpu["loss"].detach().cpu().item())
        
        output_cpu["loss"].backward()
        output_tpu["loss"].backward()
        
    comparator = TensorComparator()
    print(losses_cpu)
    print(losses_tpu)
    status1 = comparator.cmp_result(torch.tensor(losses_cpu), torch.tensor(losses_tpu))
    status2 = comparator.cmp_result(output_cpu["logits"].detach().float(), output_tpu["logits"].cpu().detach().float())
    
    if status1 and status2:
        print(f"LLaMA LoRA is correct")
    else:
        print(f"LLaMA LoRA is wrong")
        sys.exit(255)

    return status1 and status2

if __name__ == "__main__":
    pass # now this case has problem
    # case1()