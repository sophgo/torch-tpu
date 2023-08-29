import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    N = 1
    C = 4
    H = 64
    W = 64

    inp = torch.rand((N, C, H, W))
    out_cpu = inp ** 2
    inp_tpu = inp.to(device)
    out_tpu = inp_tpu ** 2
    diff = out_cpu - out_tpu.cpu()
    print("max diff: ", torch.max(torch.abs(diff)))

if __name__ == "__main__":
    case1()