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


    out_cpu = inp_cpu.sign()
    out_tpu = inp_tpu.sign()
    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    print("cpu_out")
    print(out_cpu)
    print("tpu_out")
    print(out_tpu)
    
    print (torch.max(abs(out_diff)))

if __name__ == "__main__":
    case1()