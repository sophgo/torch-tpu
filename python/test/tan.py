import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


def case1():
    a1 = torch.rand((5, 5)) - 0.5
    a1 = torch.linspace(-10, 10, 50)
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
    print(a2)
    a2.tan_()
    a2_tpu = a2_tpu.tan()
    print(a2)
    print(a2_tpu.to("cpu"))
    print((a2 - a2_tpu.to("cpu")))
    assert (a2 - a2_tpu.to("cpu")).mean()  < 1e-6
    print("pass")
    # torch.tan(a2_tpu)
    # print("origin: ", a1)
    # print("cpu : ", a2)
    # print("tpu : ", a2_tpu.cpu())


if __name__ == "__main__":
    case1()
