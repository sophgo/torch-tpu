import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():

    # and same shape
    # a1 = torch.randint(1, 10, (5,5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (5,5), dtype=torch.int)
    # a3 = a1.bitwise_and(a2)
    # a1_tpu = a1_tpu.bitwise_and(a2.to(device))
    # print("origin a1: ", a1)
    # print("origin a2: ", a2)
    # print("cpu : ", a3)
    # print("tpu : ", a1_tpu.cpu())

    # and bcast
    # a1 = torch.randint(1, 10, (1,5), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.randint(1, 10, (5,5), dtype=torch.int)
    # a3 = a1.bitwise_and(a2)
    # a1_tpu = a1_tpu.bitwise_and(a2.to(device))
    # print("origin a1: ", a1)
    # print("origin a2: ", a2)
    # print("cpu : ", a3)
    # print("tpu : ", a1_tpu.cpu())

    # # andc
    a1 = torch.randint(1, 10, (5,1), dtype=torch.int)
    a1_tpu = a1.to(device)
    a2 = torch.tensor(2, dtype=torch.int)
    a3 = a1.bitwise_and(a2)
    a1_tpu = a1_tpu.bitwise_and(a2)
    print("origin a1: ", a1)
    print("origin a2: ", a2)
    print("cpu : ", a3)
    print("tpu : ", a1_tpu.cpu())
    # andc
    # a1 = torch.randint(1, 10, (5,1), dtype=torch.int)
    # a1_tpu = a1.to(device)
    # a2 = torch.tensor(2, dtype=torch.int)
    # a3 = a1.bitwise_and(a2)
    # a1_tpu = a1_tpu.bitwise_and(a2.to(device))
    # print("origin a1: ", a1)
    # print("origin a2: ", a2)
    # print("cpu : ", a3)
    # print("tpu : ", a1_tpu.cpu())


if __name__ == "__main__":
    case1()