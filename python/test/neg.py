import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


def case1():
    a1 = torch.randn(5)
    a2 = a1.clone()
    a2_tpu = a2.to(device)
    a3 = 1
    a4 = a2.clone()

    a2.neg_()
    a2_tpu = a2_tpu.neg()
    print(abs(a2 - a2_tpu.to("cpu")).max().item())
    print(a2.size(),a2.dim())
  #  assert (a2 - a2_tpu.to("cpu") < 1e-5).all()
    print("origin: ",a1)
    print("cpu : ", a2 )
    print("tpu : ", a2_tpu.cpu())


if __name__ == "__main__":
    case1()