import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


def case1():
    
    N = 4
    C = 128
    H = 54
    W = 100

    inp_cpu = torch.rand(N, C, H, W)
    inp_tpu = copy.deepcopy(inp_cpu).to(device).to(torch.float32)

    net_cpu = torch.nn.LeakyReLU(0.1)
    net_tpu = copy.deepcopy(net_cpu).to(device).to(torch.float32)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu).to("cpu")

    diff = torch.max(abs(out_cpu - out_tpu))
    print("cpu_out:", out_cpu.flatten()[:10])
    print("tpu_out:", out_tpu.flatten()[:10])

    print("max(abs(diff)):", torch.max(abs(diff)), "\n")


if __name__ == "__main__":
    case1()