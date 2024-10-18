import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    N = 1
    C = 4
    H = 64
    W = 64

    inp = torch.ones((N, C, H, W))
    inp_tpu = inp.to(device)
    out_tpu = torch.rand_like(inp_tpu)
    print("out_tpu", out_tpu.cpu())

def randint():
    res_cpu = torch.randint(0, 1000, (5,), device='cpu')
    res_tpu = torch.randint(0, 1000, (5,), device='tpu')
    print("res cpu: ", res_cpu, ", dtype :", res_tpu.dtype)
    print("res tpu: ", res_tpu.cpu(), ", dtype :", res_tpu.dtype)
    

if __name__ == "__main__":
    case1()
    randint()