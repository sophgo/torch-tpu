import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():

    # # xor same shape
    # a1 = torch.randint(1, 10, (5,5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (5,5), dtype=torch.int)
    # cpu_res = a1.bitwise_xor(a2)
    # tpu_res = a1_tpu.bitwise_xor(a2.to(device)).cpu()

    # # xor bcast
    # a1 = torch.randint(1, 10, (2, 3, 1, 5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (5, 5), dtype=torch.int)
    # cpu_res = a1.bitwise_xor(a2)
    # tpu_res = a1_tpu.bitwise_xor(a2.to(device)).cpu()

    # # xorc
    # a1 = torch.randint(1, 10, (3,350,350), dtype=torch.int8)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(2, dtype=torch.int8)
    # cpu_res = a1.bitwise_xor(a2)
    # tpu_res = a1_tpu.bitwise_xor(a2).cpu()

    # xorc
    a1 = torch.tensor(2, dtype=torch.int8)
    a1_tpu = a1.to(device)
    a2 = torch.randint(1, 10, (3,350,350), dtype=torch.int8)
    a2_tpu = a2.to(device)
    cpu_res = a1.bitwise_xor(a2)
    tpu_res = a1_tpu.bitwise_xor(a2_tpu).cpu()

    print("origin a1: ", a1)
    print("origin a2: ", a2)
    print("cpu res : ", cpu_res)
    print("tpu res : ", tpu_res)
    print("cpu res shape : ", cpu_res.size())
    print("tpu res shape : ", tpu_res.size())
    print("max diff: ", torch.max(cpu_res - tpu_res))


if __name__ == "__main__":
    case1()