import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

device = "privateuseone:0"

def case1():
    batch = 2
    sequence = 4

    inp_cpu = torch.rand(batch, sequence, sequence)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)

    inp_cpu_triu = inp_cpu.triu(0)
    inp_tpu_triu = inp_tpu.triu(0)

    diff = torch.max(abs(inp_cpu_triu - inp_tpu_triu.to("cpu")))
    print("max(abs(diff)):", diff, "\n")

def case2():
    batch = 2
    sequence = 4

    inp_cpu = torch.rand(batch, sequence, sequence).to(torch.float16)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)

    inp_cpu_triu = inp_cpu.triu(0)
    inp_tpu_triu = inp_tpu.triu(0)

    diff = torch.max(abs(inp_cpu_triu - inp_tpu_triu.to("cpu")))
    print("max(abs(diff)):", diff, "\n")

if __name__ == "__main__":
    case1()