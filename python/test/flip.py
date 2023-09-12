import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compare_model_grad, Optimer

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"
OPT = Optimer()

def case1():
    a1 = torch.randn((2, 2))
    a2 = a1.clone()
    a2_tpu = a2.to(device)

    dims = [1]
    a2_cpu = a2.flip(dims)

    OPT.reset()
    a2_tpu = a2_tpu.flip(dims)
    OPT.dump()

    print("origin: ",a1)
    print("cpu : ", a2_cpu )
    print("tpu : ", a2_tpu.cpu())
    print("max diff : ", abs(a2_cpu - a2_tpu.to("cpu")).max().item())

if __name__ == "__main__":
    case1()