import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():
    # # lt fp
    # a1 = torch.rand((5, 6))
    # a1_tpu = a1.to(device)
    # a2 = torch.rand((5, 6))
    # a2_tpu = a2.to(device)
    # cpu_res = torch.lt(a1, a2)
    # tpu_res = torch.lt(a1_tpu, a2_tpu)

    # # lt int
    # a1 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a2_tpu = a2.to(device)
    # cpu_res = torch.lt(a1, a2)
    # tpu_res = torch.lt(a1_tpu, a2_tpu)

    # # lt c1 fp
    # a1 = torch.rand((1, 5))
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(0.5)
    # cpu_res = torch.lt(a1, a2)
    # # tpu_res = torch.lt(a1_tpu, a2)
    # tpu_res = torch.lt(a1_tpu, a2.to(device))

    # # lt c2 int
    # a1 = torch.randint(1, 10, (1, 5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(5)
    # cpu_res = torch.lt(a1, a2)
    # # tpu_res = torch.lt(a1_tpu, a2)
    # tpu_res = torch.lt(a1_tpu, a2.to(device))

    # # lt bcast1 int
    # a1 = torch.randint(1, 10, (7, 35, 23, 35), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (23, 35), dtype=torch.int)
    # cpu_res = torch.lt(a1, a2)
    # tpu_res = torch.lt(a1_tpu, a2.to(device))

    # lt bcast2 float
    a1 = torch.rand((3, 555, 35))
    a1_tpu = a1.to(device)
    a2 = torch.rand((35))
    cpu_res = torch.lt(a1, a2)
    tpu_res = torch.lt(a1_tpu, a2.to(device))

    print("a1 : ", a1)
    print("a2 : ", a2)
    print("cpu result : ", cpu_res)
    print("tpu result : ", tpu_res.cpu())
    element_size = 1
    for num in cpu_res.size():
        element_size *= num
    print("max diff : ", torch.sum(cpu_res == tpu_res.cpu()) - element_size)

if __name__ == "__main__":
    case1()