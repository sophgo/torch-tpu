import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"

def case1():
    # # le fp
    # a1 = torch.rand((5, 6))
    # a1_tpu = a1.to(device)
    # a2 = torch.rand((5, 6))
    # a2_tpu = a2.to(device)
    # cpu_res = torch.le(a1, a2)
    # tpu_res = torch.le(a1_tpu, a2_tpu)

    # # le int
    # a1 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a2_tpu = a2.to(device)
    # cpu_res = torch.le(a1, a2)
    # tpu_res = torch.le(a1_tpu, a2_tpu)

    # # le c1 fp
    # a1 = torch.rand((1, 5))
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(0.5)
    # cpu_res = torch.le(a1, a2)
    # tpu_res = torch.le(a1_tpu, a2)
    # # tpu_res = torch.le(a1_tpu, a2.to(device))

    # # le c2 int
    # a1 = torch.randint(1, 10, (5, 5, 64), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(5)
    # cpu_res = torch.le(a1, a2)
    # tpu_res = torch.le(a1_tpu, a2)
    # # tpu_res = torch.le(a1_tpu, a2.to(device))

    # # le bcast1 int
    # a1 = torch.randint(1, 10, [3, 555, 35], dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, [35], dtype=torch.int)
    # cpu_res = torch.le(a1, a2)
    # tpu_res = torch.le(a1_tpu, a2.to(device))

    # le bcast2 float
    a1 = torch.rand((3, 15, 35))
    a1_tpu = a1.to(device)
    a2 = torch.rand((15, 35))
    cpu_res = torch.le(a1, a2)
    tpu_res = torch.le(a1_tpu, a2.to(device))

    print("a1 : ", a1)
    print("a2 : ", a2)
    print("cpu result : ", cpu_res)
    print("tpu result : ", tpu_res.cpu())
    element_size = 1
    for num in cpu_res.size():
        element_size *= num
    print("max diff : ", torch.sum(cpu_res == tpu_res.cpu()) - element_size)

def case2():
    # le fp
    a1 = torch.rand((5, 6))
    a1[0][0] = 0.5
    a1_tpu = a1.to(device)
    a2 = 0.5
    cpu_res = a1 < a2
    tpu_res = a1_tpu < a2

    # # le int
    # a1 = torch.randint(1, 10, (3, 5), dtype=torch.int32)
    # a1_tpu = a1.to(device)
    # a2 = 6
    # cpu_res = a1 < a2
    # tpu_res = a1_tpu < a2

    print("a1 : ", a1)
    print("a2 : ", a2)
    print("cpu result : ", cpu_res)
    print("tpu result : ", tpu_res.cpu())
    element_size = 1
    for num in cpu_res.size():
        element_size *= num
    print("max diff : ", torch.sum(cpu_res == tpu_res.cpu()) - element_size)

if __name__ == "__main__":
    # case1()
    case2()