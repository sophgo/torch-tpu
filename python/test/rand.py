import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    N = 1
    C = 4
    H = 64
    W = 64

    inp = torch.ones((N, C, H, W))
    inp_tpu = inp.to(device)
    out_tpu = torch.rand_like(inp_tpu)
    print("out_tpu", out_tpu.cpu())

if __name__ == "__main__":
    case1()