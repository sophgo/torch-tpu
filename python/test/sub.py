import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():

    a1 = torch.randint(0, 5, (5, 5), dtype=torch.float32)
    a1_clone = a1.clone()
    a1_tpu = a1.clone().to(device)
    a2 = torch.randint(1 ,5, (5, 5), dtype=torch.float32)
    a2_clone = a2.clone()
    a2_tpu = a2.clone().to(device)


    a1.sub_(a2)
    a1_tpu.sub_(a2_tpu)
    print("***************test_sub_normal begin*****************")
    print("a1: ",a1_clone)
    print("a2 : ", a2_clone)
    print("cpu : ", a1)
    print("tpu : ", a1_tpu.cpu())
    print("***************test_sub_normal end*****************")

def case2():

    a1 = torch.randint(0, 5, (5, 5), dtype=torch.int32)
    a1_clone = a1.clone()
    a1_tpu = a1.clone().to(device)
    a2 = torch.randint(1 ,5, (1, 5), dtype=torch.int32)
    a2_clone = a2.clone()
    a2_tpu = a2.clone().to(device)


    a1.sub_(a2)
    a1_tpu.sub_(a2_tpu)
    print("***************test_sub_broadcast begin*****************")
    print("a1: ",a1_clone)
    print("a2 : ", a2_clone)
    print("cpu : ", a1)
    print("tpu : ", a1_tpu.cpu())
    print("***************test_sub_broadcast end*****************")

def case3():

    a1 = torch.randint(0, 5, (5, 5), dtype=torch.int32)
    a1_clone = a1.clone()
    a1_tpu = a1_clone.to(device)
    a2 = torch.tensor(1,dtype=torch.int32)


    a1.sub_(a2)
    a1_tpu.sub_(a2)
    print("***************test_sub_const begin*****************")
    print("a1: ",a1_clone)
    print("a2 : ", a2)
    print("cpu : ", a1)
    print("tpu : ", a1_tpu.cpu())
    print("***************test_sub_const end*****************")

if __name__ == "__main__":
    # case1()
    case2()
    # case3()