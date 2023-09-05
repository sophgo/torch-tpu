import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    a1 = torch.rand((1))
    a2 = a1.clone()
    a2_tpu = a2.to(device)

    b2 = a2.expand(2, 3)
    b2_tpu = a2_tpu.expand(2, 3)

    print("origin: ",a1)
    print("cpu : ", b2 )
    print("tpu : ", b2_tpu.cpu())
    print("max diff : ", abs(a2 - b2_tpu.cpu()).max().item())

if __name__ == "__main__":
    case1()