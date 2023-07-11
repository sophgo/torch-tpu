import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    t = torch.randn(1, 3)
    t_tpu=t.to(device)
    print("origin: ",t)

    t1 = torch.randn(1,3)
    t2 = torch.randn(1, 3)
    t.addcdiv_(t1, t2, value=0.1)
    t_tpu.addcdiv_(t1.to(device),t2.to(device),value=0.1)

    print("cpu : ", t )
    print("tpu : ", t_tpu.cpu())


if __name__ == "__main__":
    case1()