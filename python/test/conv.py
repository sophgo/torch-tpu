import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import compare_model_grad

import torch_tpu
torch.manual_seed(1000)

def case1(use_fp16=False):
    """
    conv backward 
    """
    device = "tpu"
    B = 64
    C = 3
    H = 224
    W = 224
    IC = 3
    OC = 64
    K = 3

    inp_cpu = torch.rand((B,C,H,W))
    inp_tpu = copy.deepcopy(inp_cpu)
    
    inp_cpu.requires_grad = True
    inp_tpu = inp_tpu.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()
    inp_tpu.requires_grad = True

    net_cpu = nn.Conv2d(in_channels=IC, out_channels=OC, kernel_size=K)
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device)
    if use_fp16: net_tpu = net_tpu.half()

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu)
    diff = inp_cpu - inp_tpu.cpu()
    print(torch.max(abs(diff)))

    grad_o = torch.ones(out_cpu.shape)
    grad_o_tpu = grad_o.to(device)
    if use_fp16: grad_o_tpu = grad_o_tpu.half()

    out_cpu.backward(grad_o)
    out_tpu.backward(grad_o_tpu)

    diff = inp_cpu.grad - inp_tpu.grad.cpu()
    print(torch.max(abs(diff)))
    compare_model_grad(net_cpu, net_tpu)

if __name__ == "__main__":
    case1(use_fp16 = False)