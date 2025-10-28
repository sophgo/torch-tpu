import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)

device = "tpu:0"
# device = "cpu"

def case1():
    """
    silu fp16
    """
    batch = 2
    sequence = 1024
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device).to(torch.float16)

    net_cpu = nn.SiLU()
    net_tpu = copy.deepcopy(net_cpu).to(device).to(torch.float16)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu).to("cpu")

    diff = torch.max(abs(out_cpu - out_tpu))
    print("cpu_out:", out_cpu.flatten()[:10])
    print("tpu_out:", out_tpu.flatten()[:10])

    print("max(abs(diff)):", torch.max(abs(diff)), "\n")

def case2():
    """
    silu fp32
    """
    # device = "cpu"
    batch = 64 
    C     = 512
    sequence = 20
    hidden_size = 20 #262144 # 3,6,7

    inp_cpu = torch.rand(batch, C, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device)
    inp_tpu.requires_grad = True
    inp_cpu.requires_grad = True

    grad_cpu = torch.rand((batch * 2, C, sequence, hidden_size))
    grad_tpu = grad_cpu.to(device)
    grad_tpu = grad_tpu[::2,...]
    grad_cpu = grad_cpu[::2,...]
    net_cpu = nn.SiLU()
    net_tpu = net_cpu.to(device)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu)

    out_cpu.backward(grad_cpu)
    out_tpu.backward(grad_tpu)

    diff      = abs(out_cpu - out_tpu.cpu())
    diff_grad = abs(inp_cpu.grad - inp_tpu.grad.cpu())
    print(torch.max(diff))
    print(torch.max(diff_grad))
    import pdb; pdb.set_trace()

def case3():
    """
    silu bfp16
    """
    N = 4
    C = 10
    H = 512
    W = 64
    inp_cpu = torch.rand(N, C, H, W)
    inp_tpu = copy.deepcopy(inp_cpu).to(device).to(torch.bfloat16)

    net_cpu = nn.SiLU()
    net_tpu = copy.deepcopy(net_cpu).to(device).to(torch.bfloat16)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu).to("cpu")

    diff = torch.max(abs(out_cpu - out_tpu))
    print("cpu_out:", out_cpu.flatten()[:10])
    print("tpu_out:", out_tpu.flatten()[:10])

    print("max(abs(diff)):", torch.max(abs(diff)), "\n")
    
if __name__ == "__main__":
    #case1()
    case2()
    #case3()