import torch
import torch.nn as nn
import copy
import os,sys
from top_utest import TensorComparator, set_bacis_info
import pkg_resources
from pkg_resources import  DistributionNotFound, VersionConflict
import torch_tpu

torch.manual_seed(10086)
torch.set_printoptions(precision=6)
device = "tpu:0"

# now peft 0.12.0 will cause problem. So we fix the version of transformers and peft.
for package, version in {"transformers": "4.41.2", "peft": "0.11.1"}.items():
    try:
        if pkg_resources.get_distribution(package).version != version:
            raise VersionConflict
    except (DistributionNotFound, VersionConflict):
        os.system(f"PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple python3 -m pip install {package}=={version}")

os.environ["CMODEL_FAST_EXEC"]="1"

import transformers
from transformers import LlamaForCausalLM, LlamaConfig
from peft import LoraConfig, TaskType, get_peft_model
import peft
from torch_tpu.tpu.custom_op.lora_matmul import LoraMatmulBlock

# temporary fix for torch.arange
def arange_wrapper(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret.dtype == torch.int64:
            return ret.to(torch.int32)
    return wrapper
torch.arange = arange_wrapper(torch.arange)

# temporary fix for cross_entropy
def ce_wrapper(func):
    def wrapper(input, tensor, weight=None, *args, **kwargs):
        input_cpu = input.cpu()
        tensor_cpu = tensor.cpu()
        if weight is not None:
            weight_cpu = weight.cpu()
        else:
            weight_cpu = None
        return func(input_cpu, tensor_cpu, weight_cpu, *args, **kwargs).to(input.device)
    return wrapper

torch.nn.functional.cross_entropy = ce_wrapper(torch.nn.functional.cross_entropy)

llama_config_dict = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1024, 
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 1024,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 2,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "use_cache": True,
    "vocab_size": 2048
    }

lora_config_dict = {
    "r": 8,
    "target_modules": ["down_proj","v_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "use_dora": False,
    "use_rslora": False
    }

features_out = {}
def get_features(name):
    def hook_fn( module, grad_input, grad_output):
        print(f"Hook called for module: {module}")
        features_out[name]=(grad_input[0].data.cpu())
    return hook_fn

def hook_fn_forward( module, inp):
    input0 = inp[0]
    input0.requires_grad_(True)
    return tuple(input0)

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

    lora_model_cpu = lora_model.float()
    lora_model_tpu = copy.deepcopy(lora_model).to(device)
    
    #plan A: call torch_tpu api to replace lora block
    from torch_tpu.tpu.custom_op import lora_matmul
    lora_matmul.create_and_replace(lora_model_tpu, lora_config)
    
    #------------------plan B: replace lora block manually, e.g. v_proj layer-------------------------
    # for i in range(llama_config_dict["num_hidden_layers"]):
    #     v_proj = lora_model_tpu.model.model.layers[i].self_attn.v_proj
    #     assert(isinstance(v_proj, peft.tuners.lora.Linear)) 
    #     newlora_block = LoraMatmulBlock(llama_config_dict['hidden_size'],llama_config_dict['hidden_size'], v_proj.weight.data.t().contiguous().half().to(device),
    #                                    v_proj.lora_A['default'].weight.data.t().contiguous().half().to(device),
    #                                    v_proj.lora_B['default'].weight.data.t().contiguous().half().to(device),
    #                                   rank=lora_config_dict['r'],alpha=lora_config_dict['lora_alpha'],
    #                                   dropout_rate=lora_config_dict['lora_dropout']).to(device)
    #     lora_model_tpu.model.model.layers[i].self_attn.v_proj = newlora_block
    #------------------replace lora block manually-------------------------
    hook1 = lora_model_cpu.model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(hook_fn_forward)
    hook2 = lora_model_tpu.model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(hook_fn_forward)
    
    hook3 = lora_model_cpu.model.model.layers[0].self_attn.q_proj.register_backward_hook(get_features("cpu_backward"))
    hook4 = lora_model_tpu.model.model.layers[0].self_attn.q_proj.register_backward_hook(get_features("tpu_backward"))

    seq_len = 640
    comparator = TensorComparator()
    
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
    
    
    output_cpu["loss"].backward(retain_graph = True)
    output_tpu["loss"].backward(retain_graph = True)


    status1 = comparator.cmp_result(output_cpu["logits"].detach(), output_tpu["logits"].detach().cpu().float())
    status2 = comparator.cmp_result(features_out["cpu_backward"], features_out["tpu_backward"])
    
    if status1 and status2:
        print(f"LLaMA LoRA is correct")
    else:
        print(f"LLaMA LoRA is wrong, loss comparision: {status1}, output comparision: {status2}")
        sys.exit(255)

    hook1.remove()
    hook2.remove()
    hook3.remove()
    hook4.remove()
    return status1 and status2

if __name__ == "__main__":
    case1()
