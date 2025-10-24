import torch
from nnmoduletools.module_debugger import register_hook, combine_npz
import torch_tpu
device = "tpu"
from torch_tpu.tpu.custom_op.llama_attn_qkv import fuse_llama_attn_qkv
from transformers.models.llama.modeling_llama import LlamaAttention
from top_utest import TensorComparator
from transformers import LlamaConfig

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import os, copy

#Fix the version of transformers
for package, version in {"transformers": "4.41.2"}.items():
    try:
        if pkg_resources.get_distribution(package).version != version:
            raise VersionConflict
    except (DistributionNotFound, VersionConflict):
        os.system(f"PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple python3 -m pip install {package}=={version}")

os.environ["CMODEL_FAST_EXEC"]="1"

def cmp_weight_tensor(args_cpu, args_tpu):
    status = True
    for name, _ in args_cpu.items():
        print(name)
        status = status and torch.allclose(args_cpu[name], args_tpu[name].float(), atol=1e-4)
        print(args_cpu[name][3,0:20])
        print('tpu:')
        print(args_tpu[name].float()[3,0:20])
    return status

if __name__ == "__main__":
    torch.manual_seed(10086)

    config = LlamaConfig()
    #test GQA
    config.num_key_value_heads = 4
    config.num_attention_heads = 32
    llama_model = LlamaAttention(config, 0)
    model_cpu = copy.deepcopy(llama_model)
    model_tpu = copy.deepcopy(llama_model).half().to(device)

    bsz = 8
    q_len = 512

    hidden_states_cpu = torch.randn(bsz, q_len, config.hidden_size)
    hidden_states_tpu = copy.deepcopy(hidden_states_cpu).half().to(device)

    position_ids_cpu = torch.randint(0, config.max_position_embeddings, (1, q_len))
    position_ids_tpu = copy.deepcopy(position_ids_cpu).to(device)
    mask = torch.randint(0, 2, (bsz, 1, q_len, q_len))
    mask = torch.where(mask >= 1, torch.tensor(0.), torch.tensor(-float('inf'), dtype=torch.float32))
    mask_tpu = mask.half().to(device)

    output_cpu = model_cpu(hidden_states_cpu, attention_mask=mask, position_ids=position_ids_cpu)
    fuse_llama_attn_qkv()
    cpu_qkv_weight = {
        'q_proj': model_cpu.q_proj.weight.data,
        'k_proj': model_cpu.k_proj.weight.data,
        'v_proj': model_cpu.v_proj.weight.data,
        'o_proj': model_cpu.o_proj.weight.data,
    }
    tpu_qkv_weight = {
        'q_proj': model_tpu.q_proj.weight.data.cpu(),
        'k_proj': model_tpu.k_proj.weight.data.cpu(),
        'v_proj': model_tpu.v_proj.weight.data.cpu(),
        'o_proj': model_tpu.o_proj.weight.data.cpu(),
    }
    assert(cmp_weight_tensor(cpu_qkv_weight, tpu_qkv_weight))
    print("init success")

    comparator = TensorComparator()
    output_cpu = output_cpu[0]

    output_tpu = model_tpu(hidden_states_tpu, attention_mask= mask_tpu, position_ids=position_ids_tpu)[0]
    status_fwd = comparator.cmp_result(output_cpu.detach(), output_tpu.detach().cpu().float())

    print("======backward========")
    grad_output_cpu = (torch.randn(bsz, q_len, config.hidden_size))*1000
    grad_output_tpu = copy.deepcopy(grad_output_cpu).half().to(device)

    output_tpu.backward(grad_output_tpu)
    output_cpu.backward(grad_output_cpu)

    status_q = comparator.cmp_result(model_cpu.q_proj.weight.grad.detach(), model_tpu.q_proj.weight.grad.detach().cpu().float())
    status_k = comparator.cmp_result(model_cpu.k_proj.weight.grad.detach(), model_tpu.k_proj.weight.grad.detach().cpu().float())
    status_v = comparator.cmp_result(model_cpu.v_proj.weight.grad.detach(), model_tpu.v_proj.weight.grad.detach().cpu().float())
    status_o = comparator.cmp_result(model_cpu.o_proj.weight.grad.detach(), model_tpu.o_proj.weight.grad.detach().cpu().float())
    print(f"fwd: {status_fwd}")
    print(f"bwd: q {status_q},k {status_k},v {status_v},o {status_o}")