import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)

def case1(use_fp16):
    """
    gelu backward
    """
    device = "privateuseone"
    batch = 8
    sequence = 1024
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu)
    #inp_cpu.retain_grad = True
    # inp_cpu.requires_grad = True

    inp_tpu = inp_tpu.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()
    #inp_tpu.retain_grad = True
    # inp_tpu.requires_grad = True

    out_cpu = inp_cpu.fill_(1)
    out_tpu = inp_tpu.fill_(1)
    if use_fp16: inp_tpu = out_tpu.half()

    diff = out_cpu - out_tpu.cpu()
    print("diff: ", torch.max(abs(diff)))

if __name__ == "__main__":
    case1(use_fp16=1)