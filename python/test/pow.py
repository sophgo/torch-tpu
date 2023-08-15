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

def case2():
    a1 = torch.rand((5,5))*2
    a2 = a1.clone()
    a2_tpu = a2.to(device)
    a3 = torch.randn((5,5))*2

    a1.pow_(a3)
    a2_tpu = a2_tpu.pow(a3.to(device))
    # assert (a2 - a2_tpu.to("cpu") < 1e-5).all()
    print("a1: ",a1)
    print("a3: ",a3)
    print("cpu : ", a1 )
    print("tpu : ", a2_tpu.cpu())
    print("max diff: ", abs(a1 - a2_tpu.to("cpu")).max().item())


if __name__ == "__main__":
    case1()
    case2()