import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=8)
device = "privateuseone:0"

def case1():
    a1 = torch.rand(2, 35, 4, 33)
    a1_tpu = a1.to(device)
    cpu_res = torch.prod(a1, 0, False)
    tpu_res = torch.prod(a1_tpu, 0, False).cpu()

    # print("origin : ", a1)
    print("cpu result : ", cpu_res[0,:10])
    print("tpu result : ", tpu_res[0,:10])
    print("cpu shape : ", cpu_res.size())
    print("tpu shape : ", tpu_res.size())
    print("max diff : ", torch.max(torch.div(torch.abs(cpu_res - tpu_res), cpu_res)))

if __name__ == "__main__":
    case1()