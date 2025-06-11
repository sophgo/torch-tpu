import torch
import torch.nn as nn
import copy
import os,sys
from top_utest import TensorComparator, set_bacis_info
import pkg_resources
from pkg_resources import  DistributionNotFound, VersionConflict
import torch_tpu
import os
os.environ["CMODEL_FAST_EXEC"]="1"

device = "tpu:0"

# now peft 0.12.0 will cause problem. So we fix the version of transformers and peft._
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


# temporary fix for torch.arange_
def arange_wrapper(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if ret.dtype == torch.int64:
            return ret.to(torch.int32)
        else:
            return ret
    return wrapper
torch.arange = arange_wrapper(torch.arange)


# temporary fix for cross_entropy_
def ce_wrapper(func):
    def wrapper(input, tensor, weight=None, *args, **kwargs):
        input_cpu = input.cpu()
        tensor_cpu = tensor.cpu()
        if(tensor_cpu.dtype != torch.int64 ):
            tensor_cpu = tensor.cpu().to(torch.int64)
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

tensors = {}
def hook_backward(name: str, index: int, tpu: bool = False):
    def hook_fn(module, grad_input, grad_output):
        input = grad_input[index]
        tensors[name + "_grad"] = input
        tensors[name + "_output"] = grad_output[0]
    return hook_fn

def hook_grad(module, input):
    input0 = input[0]
    input0.requires_grad = True
    return input0

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
    lorab = lora_model_cpu.model.model.layers[0].self_attn.v_proj.lora_B['default'].weight
    lorab.data = torch.randn(lorab.shape, dtype=torch.float32)
    lora_model_tpu = copy.deepcopy(lora_model).half().to(device)

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

    hook0 = lora_model_cpu.model.model.layers[0].self_attn.v_proj.register_forward_pre_hook(hook_grad)
    hook0 = lora_model_tpu.model.model.layers[0].self_attn.v_proj.register_forward_pre_hook(hook_grad)
    hook1 = lora_model_cpu.model.model.layers[0].self_attn.v_proj.register_full_backward_hook(hook_backward("cpu_v_proj_input", 0))
    hook2 = lora_model_tpu.model.model.layers[0].self_attn.v_proj.register_full_backward_hook(hook_backward("tpu_v_proj_input", 0, True))

    seq_len = 256
    comparator = TensorComparator()

    input_ids = torch.randint(0, llama_config_dict["vocab_size"], (1, seq_len))
    input_labels = torch.randint(1000, llama_config_dict["vocab_size"], (1, seq_len))

    input_ids_tpu = copy.deepcopy(input_ids).to(device)
    input_labels_tpu = copy.deepcopy(input_labels).to(device)

    input_ids_cpu = input_ids
    input_labels_cpu = input_labels

    lora_model_cpu.train()
    lora_model_tpu.train()

    output_cpu = lora_model_cpu(input_ids_cpu, labels=input_labels_cpu)
    output_tpu = lora_model_tpu(input_ids_tpu, labels=input_labels_tpu)

    output_cpu["loss"].backward()
    output_tpu["loss"].backward()

    # Get gradients after backward pass
    cpu_loraA_grad = lora_model_cpu.model.model.layers[0].self_attn.v_proj.lora_A['default'].weight.grad
    cpu_loraB_grad = lora_model_cpu.model.model.layers[0].self_attn.v_proj.lora_B['default'].weight.grad
    cpu_lora_input_grad = tensors["cpu_v_proj_input_grad"] #v_proj input grad

    tpu_loraA_grad = lora_model_tpu.model.model.layers[0].self_attn.v_proj.loraA.grad
    tpu_loraB_grad = lora_model_tpu.model.model.layers[0].self_attn.v_proj.loraB.grad
    tpu_lora_input_grad = tensors["tpu_v_proj_input_grad"] #v_proj input grad

    print("===========fwd & bwd===========")
    status1 = comparator.cmp_result(output_cpu["logits"].detach(), output_tpu["logits"].detach().cpu().float())
    status2 = comparator.cmp_result(cpu_loraA_grad, tpu_loraA_grad.cpu().float())
    status3 = comparator.cmp_result(cpu_loraB_grad, tpu_loraB_grad.cpu().float())
    status4 = comparator.cmp_result(cpu_lora_input_grad, tpu_lora_input_grad.cpu().float())
    print(f"{status1}, {status2}, {status3}, {status4}")
    status2 = status2 and status3 and status4
    if status1 and status2:
        print(f"LLaMA LoRA is correct")
    else:
        print(f"LLaMA LoRA is wrong, loss comparision: {status1}, gradient comparision: {status2}")
        sys.exit(255)

    return status1 and status2

if __name__ == "__main__":
    case1()
