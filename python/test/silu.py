import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)

device = "privateuseone:0"
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
    batch = 1
    sequence = 1024
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device).to(torch.float32)

    net_cpu = nn.SiLU()
    net_tpu = copy.deepcopy(net_cpu).to(device).to(torch.float32)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu).to("cpu")

    diff = torch.max(abs(out_cpu - out_tpu))
    print("cpu_out:", out_cpu.flatten()[:10])
    print("tpu_out:", out_tpu.flatten()[:10])

    print("max(abs(diff)):", torch.max(abs(diff)), "\n")

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
    case1()
    case2()
    case3()