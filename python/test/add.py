import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():

    a1 = torch.tensor(1, dtype=torch.int)
    a2 = torch.tensor(1.0, dtype=torch.float)
    a2_tpu = a2.to(device)
    a3 = 1
    a4 = torch.tensor(1, dtype=torch.int)


    # tensor add tensor
    # a2.add_(a1)
    # a2_tpu.add_(a1.to(device))
    # print("origin: ",a1)
    # print("cpu : ", a2 )
    # print("tpu : ", a2_tpu.cpu())

    # tensor add scalar
    # a2.add_(a3)
    # a2_tpu.add_(float(a3))
    # print("origin: ",a1)
    # print("cpu : ", a2 )
    # print("tpu : ", a2_tpu.cpu())

    # broadcast add
    a2.add_(a4)
    a2_tpu.add_(a4.to(device))
    print("origin: ",a1)
    print("cpu : ", a2 )
    print("tpu : ", a2_tpu.cpu())

def case_addbcast():
    N = 1
    C = 4096
    H = 512
    a = torch.randn(N, C, H)
    b = torch.randn(H)
    a_tpu = a.to(device)
    b_tpu = b.to(device)

    o_t = a_tpu + b_tpu
    o = a + b

    diff = o - o_t.cpu()
    print(torch.max(torch.abs(diff)))


if __name__ == "__main__":
    #case1()
    case_addbcast()