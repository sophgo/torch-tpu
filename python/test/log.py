import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


def case1():
    a1 = torch.rand((30, 2500, 500))
    a2 = a1.clone()
    a2_tpu = a2.to(device)
    a3 = 1
    a4 = a2.clone()

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
    # print(a2)
    a2.log_()
    a2_tpu = a2_tpu.log()
    # print(a2)
    # print(a2_tpu.to("cpu"))
    print(abs(a2 - a2_tpu.to("cpu")).max().item())
    assert (a2 - a2_tpu.to("cpu") < 1e-5).all()
    print("pass")
    # torch.log(a2_tpu)
    # print("origin: ", a1)
    # print("cpu : ", a2)
    # print("tpu : ", a2_tpu.cpu())


if __name__ == "__main__":
    case1()
