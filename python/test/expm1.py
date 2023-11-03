import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


def case1():
    a1 = torch.randn((5,5))
    a2 = a1.clone()
    a2_tpu = a2.to(device)

    a2.expm1_()
    a2_tpu = a2_tpu.expm1()

    print("origin: ",a1)
    print("cpu : ", a2 )
    print("tpu : ", a2_tpu.cpu())
    print("max diff : ", abs(a2 - a2_tpu.to("cpu")).max().item())

if __name__ == "__main__":
    case1()