import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
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

def case3():
    """
    test y = x^2, x^3, x^4. Implement Pow using multiplication
    
    dtype: fp32 & fp16 & bfp16
    """
    N = 4
    C = 4
    H = 64
    W = 500
    exponent = 4
    inp = torch.rand((N, C, H, W))
    out_cpu = inp ** exponent
    inp_tpu = inp.to(device)
    out_tpu = inp_tpu ** exponent
    out = out_tpu.cpu()
    diff = out_cpu - out
    print("max diff(fp32): ", torch.max(torch.abs(diff)))

    inp_tpu0 = inp.to(device).to(torch.float16)
    out_tpu0 = inp_tpu0 ** exponent
    out0 = out_tpu0.cpu()
    diff = out_cpu - out0
    print("max diff(fp16): ", torch.max(torch.abs(diff)))

    inp_tpu1 = inp.to(device).to(torch.bfloat16)
    out_tpu1 = inp_tpu1 ** exponent
    out1 = out_tpu1.cpu()
    diff = out_cpu - out1
    print("max diff(bfp16): ", torch.max(torch.abs(diff)))

if __name__ == "__main__":
    # case1()
    # case2()
    case3()
