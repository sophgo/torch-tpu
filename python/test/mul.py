import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    H = 386
    W = 256
    aa = torch.ones((5,5))
    
    a = torch.norm(aa, 2.0)
    inp = torch.randn((H, W))
    inp_tpu = inp.to(device)

    inp.mul_(a)
    inp_tpu.mul_(a.to(device))
    print("cpu : ", inp )
    print("tpu : ", inp_tpu.cpu())


if __name__ == "__main__":
    case1()