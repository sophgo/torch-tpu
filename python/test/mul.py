import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

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

    cst = 2
    inp = torch.tensor((2,2),dtype=torch.int)
    inp_tpu = inp.to(device)
    inp.mul_(cst)
    inp_tpu.mul_(cst)
    print("cpu : ", inp )
    print("tpu : ", inp_tpu.cpu())

def case2():
    B = 6
    S = 1024
    H = 768

    a = torch.randn((B,S,H))
    b = torch.randn((S,H))

    # # test fp16
    # a_tpu = a.half().to(device)
    # b_tpu = b.half().to(device)

    a_tpu = a.to(device)
    b_tpu = b.to(device)

    o = a * b
    o_tpu = a_tpu * b_tpu

    diff = o - o_tpu.cpu()
    print(torch.max(torch.abs(diff)))



if __name__ == "__main__":
    case2()
    # case1()
