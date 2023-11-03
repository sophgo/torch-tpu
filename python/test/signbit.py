import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)

def case1():
    device = "privateuseone"
    sequence = 50257
    hidden_size = 768

    inp_cpu = torch.randn(sequence, hidden_size)
    inp_tpu = inp_cpu.to(device).half()


    out_cpu = torch.signbit(inp_cpu)
    out_tpu = torch.signbit(inp_tpu)
    print("origin",inp_cpu)
    print("cpu_out")
    print(out_cpu)
    print("tpu_out")
    print(out_tpu)
    

if __name__ == "__main__":
    print("case1:")
    case1()
   