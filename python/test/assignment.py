from turtle import forward
import torch
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn as nn
import copy
from utils import Optimer

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

device = "privateuseone"
optimer = Optimer("../../libtorch_plugin/build/liblibtorch_plugin.so")


def case1():
    """a = b
    """
    in_fea = 10
    out_fea = 5
    a = torch.empty(in_fea, out_fea)
    a_tpu = a.to(device)
    b = torch.empty(in_fea, out_fea)
    b_tpu = b.to(device)
    print("id a :", id(a), ", id b: ", id(b))
    print("id tpu_a : ", id(a_tpu), ", id tpu_b : ", id(b_tpu))
    a = b
    a_tpu = b_tpu
    print("===== after assignment =====")
    print("id a :", id(a), ", id b: ", id(b))
    print("id tpu_a : ", id(a_tpu), ", id tpu_b : ", id(b_tpu))

def case2():
    """a = b
    """
    in_fea = 10
    out_fea = 5
    a = torch.empty(in_fea, out_fea)
    b = torch.empty(in_fea, out_fea)
    a = b
    a_tpu = a.to(device)
    b_tpu = b.to(device)

    print("id a :", id(a), ", id b: ", id(b))
    print("id tpu_a : ", id(a_tpu), ", id tpu_b : ", id(b_tpu))

def case3():
    """
    """
    in_fea = 10
    out_fea = 5
    a = Parameter(torch.empty(in_fea, out_fea))
    b = Parameter(torch.empty(in_fea, out_fea))
    a = b
    a_tpu = a.to(device)
    b_tpu = b.to(device)
    print("===== after assignment =====")
    print("id a :", id(a), ", id b: ", id(b))
    print("id tpu_a : ", id(a_tpu), ", id tpu_b : ", id(b_tpu))

def case4():
    in_fea = 10
    out_fea = 5
    net_a = nn.Linear(in_fea, out_fea,bias=False)
    net_b = nn.Linear(in_fea, out_fea, bias=False)
    print("id a :", net_a.weight.storage().data_ptr(), ", id b: ", net_b.weight.storage().data_ptr())

    net_a.weight = net_b.weight

    net_a_tpu = copy.deepcopy(net_a)
    net_b_tpu = copy.deepcopy(net_b)
    net_a_tpu =  net_a_tpu.to(device)
    net_b_tpu =  net_b_tpu.to(device)
    print("===== after assignment =====")
    print("id a :", net_a.weight.storage().data_ptr(), ", id b: ", net_b.weight.storage().data_ptr())
    print("device a :", net_a.weight.device, ", device b: ", net_b.weight.device)

    print("id tpu_a : ", net_a_tpu.weight.storage().data_ptr(), ", id tpu_b : ", net_b_tpu.weight.storage().data_ptr())
    print("device tpu_a : ", net_a_tpu.weight.device, ", device tpu_b : ", net_b_tpu.weight.device)

def case5():
    in_fea = 10
    out_fea = 5
    net_a = nn.Linear(in_fea, out_fea,bias=False)
    net_b = nn.Linear(in_fea, out_fea, bias=False)
    net_a.weight = net_b.weight

    net_a_tpu = copy.deepcopy(net_a)
    net_b_tpu = copy.deepcopy(net_b)
    net_a_tpu =  net_a_tpu.to(device)
    net_b_tpu =  net_b_tpu.to(device)
    print("===== after assignment =====")
    print("id a :", id(net_a.weight), ", id b: ", id(net_b.weight))
    print("id tpu_a : ", id(net_a_tpu.weight), ", id tpu_b : ", id(net_b_tpu.weight))

def case6():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 5, bias=False)
            self.l2 = nn.Linear(10, 5, bias=False)
            self.l1.weight = self.l2.weight

        def forward(self, x):
           pass
    net = Net()

    net_tpu = copy.deepcopy(net)
    net_tpu.to(device)
    print("===== after assignment =====")
    print("id a :", id(net.l1.weight), ", id b: ", id(net.l2.weight))
    print("id tpu_a : ", id(net_tpu.l1.weight), ", id tpu_b : ", id(net_tpu.l2.weight))
    print("a ptr : ", net_tpu.weight.storage().data_ptr(), ", id tpu_b : ", net_tpu.weight.storage().data_ptr())

if __name__ == "__main__":
    # case1()
    # case2()
    # case3()
    case6()