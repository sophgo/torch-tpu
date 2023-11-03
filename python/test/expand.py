import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Optimer

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"
OPT = Optimer()

def case1():
    a1 = torch.rand(2, 1, 4)
    a2 = a1.clone()
    a2_tpu = a2.to(device)

    b2 = a2.expand(2, 2, 4)
    OPT.reset()
    b2_tpu = a2_tpu.expand(2, 2, 4)
    OPT.dump()

    print("origin: ",a1)
    print("cpu : ", b2 )
    print("tpu : ", b2_tpu.cpu())
    print("max diff : ", abs(a2 - b2_tpu.cpu()).max().item())

if __name__ == "__main__":
    case1()