import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():
    # gt fp
    a1 = torch.rand((5, 6))
    a1_tpu = a1.to(device)
    a2 = torch.rand((5, 6))
    a2_tpu = a2.to(device)
    cpu_res = torch.gt(a1, a2)
    tpu_res = torch.gt(a1_tpu, a2_tpu)

    # # gt int
    # a1 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a2_tpu = a2.to(device)
    # cpu_res = torch.gt(a1, a2)
    # tpu_res = torch.gt(a1_tpu, a2_tpu)

    # # gt c1 fp
    # a1 = torch.rand((1, 5))
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(0.5)
    # cpu_res = torch.gt(a1, a2)
    # tpu_res = torch.gt(a1_tpu, a2)
    # # tpu_res = torch.gt(a1_tpu, a2.to(device))

    # # gt c2 int
    # a1 = torch.randint(1, 10, (1, 5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(5)
    # cpu_res = torch.gt(a1, a2)
    # tpu_res = torch.gt(a1_tpu, a2)
    # # tpu_res = torch.gt(a1_tpu, a2.to(device))

    print("a1 : ", a1)
    print("a2 : ", a2)
    print("cpu result : ", cpu_res)
    print("tpu result : ", tpu_res)

if __name__ == "__main__":
    case1()