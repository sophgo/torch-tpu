import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():
    # # not equal fp
    # a1 = torch.tensor([[2, 5, 2, 4, 3],
    #                    [9, 7, 1, 3, 9],
    #                    [3, 8, 3, 9, 9],
    #                    [3, 3, 4, 6, 5],
    #                    [3, 4, 3, 5, 3]], dtype=torch.float)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor([[2, 5, 2, 4, 5],
    #                    [9, 7, 1, 6, 9],
    #                    [3, 8, 5, 9, 9],
    #                    [3, 8, 4, 6, 5],
    #                    [2, 4, 3, 5, 3]], dtype=torch.float)
    # a2_tpu = a2.to(device)
    # cpu_res = torch.ne(a1, a2)
    # tpu_res = torch.ne(a1_tpu, a2_tpu)

    # # not equal int
    # a1 = torch.randint(1, 10, (1, 75), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (1, 75), dtype=torch.int)
    # a2_tpu = a2.to(device)
    # cpu_res = torch.ne(a1, a2)
    # tpu_res = torch.ne(a1_tpu, a2_tpu)

    # # not equal c1 fp
    # a1 = torch.rand((5, 5))
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(0.4217)
    # cpu_res = torch.ne(a1, a2)
    # # tpu_res = torch.ne(a1_tpu, a2)
    # tpu_res = torch.ne(a1_tpu, a2.to(device))

    # # not equal c2 int
    # a1 = torch.randint(1, 10, (5, 5, 5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(5)
    # cpu_res = torch.ne(a1, a2)
    # tpu_res = torch.ne(a1_tpu, a2)
    # # tpu_res = torch.ne(a1_tpu, a2.to(device))

    # not equal bcast1 int
    a1 = torch.randint(1, 10, (33, 15, 35), dtype=torch.int)
    a1_tpu = a1.to(device)
    a2 = torch.randint(1, 10, (15, 35), dtype=torch.int)
    cpu_res = torch.ne(a1, a2)
    tpu_res = torch.ne(a1_tpu, a2.to(device))

    # # not equal bcast2 float
    # a1 = torch.rand((3, 555, 35))
    # a1_tpu = a1.to(device)
    # a2 = torch.rand((555, 35))
    # cpu_res = torch.ne(a1, a2)
    # tpu_res = torch.ne(a1_tpu, a2.to(device))

    print("a1 : ", a1)
    print("a2 : ", a2)
    print("cpu result : ", cpu_res)
    print("tpu result : ", tpu_res.cpu())
    element_size = 1
    for num in cpu_res.size():
        element_size *= num
    print("max diff : ", torch.sum(cpu_res == tpu_res.cpu()) - element_size)

def case2():
    # not equal fp
    a1 = torch.tensor([[2, 5, 2, 4, 3],
                       [9, 7, 1, 3, 9],
                       [3, 8, 3, 9, 9],
                       [3, 3, 4, 6, 5],
                       [3, 4, 3, 5, 3]], dtype=torch.float)
    a1_tpu = a1.to(device)
    a2 = 9.0
    cpu_res = a1 != a2
    tpu_res = a1_tpu != a2

    # # not equal int
    # a1 = torch.randint(1, 10, (1, 75), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = 6
    # cpu_res = a1 != a2
    # tpu_res = a1_tpu != a2


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